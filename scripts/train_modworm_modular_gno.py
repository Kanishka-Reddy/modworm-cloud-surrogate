#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zarr
import matplotlib.pyplot as plt

# --- 1. Modular Model Components ---

class NeuralGNOBlock(nn.Module):
    """
    Message passing over gap-junction and synaptic connectome.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, adj_gap, adj_syn):
        super().__init__()
        self.register_buffer("adj_gap", torch.from_numpy(adj_gap).float())
        self.register_buffer("adj_syn", torch.from_numpy(adj_syn).float())
        
        self.node_init = nn.Linear(in_dim, hidden_dim)
        
        # 1 Layer of typed message passing
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self": nn.Linear(hidden_dim, hidden_dim),
                "gap": nn.Linear(hidden_dim, hidden_dim),
                "syn": nn.Linear(hidden_dim, hidden_dim)
            }) for _ in range(1)
        ])
        
        self.out_head = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, feedback=None):
        # x: [B, 279, in_dim]
        h = self.node_init(x)
        if feedback is not None:
            h = h + feedback
            
        for layer in self.layers:
            h_self = layer["self"](h)
            
            # Message passing: [B, N, N] @ [B, N, D] -> [B, N, D]
            # Since adj is constant, we can use it as [N, N]
            h_gap = torch.matmul(self.adj_gap, h)
            h_gap = layer["gap"](h_gap)
            
            h_syn = torch.matmul(self.adj_syn, h)
            h_syn = layer["syn"](h_syn)
            
            h = F.relu(self.ln(h_self + h_gap + h_syn))
            
        return self.out_head(h)

class MuscleBlock(nn.Module):
    """
    Map neural node embeddings to muscle channels and predict delta muscle.
    """
    def __init__(self, neural_hidden_dim, muscle_dim, muscle_map):
        super().__init__()
        # muscle_map is [96, 279]. We aggregate to [48, 279]
        # Assuming 48 pairs of (dorsal, ventral).
        m48_map = (muscle_map[::2] + muscle_map[1::2]) / 2.0
        self.register_buffer("m_map", torch.from_numpy(m48_map).float())
        
        self.net = nn.Sequential(
            nn.Linear(neural_hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h_neural, m_t):
        # h_neural: [B, 279, D]
        # Project neural embeddings to muscle space
        # [B, 48, 279] @ [B, 279, D] -> [B, 48, D]
        m_feats = torch.matmul(self.m_map, h_neural)
        
        # Combine with current muscle state: [B, 48, D+1]
        m_t_expanded = m_t.unsqueeze(-1) # [B, 48, 1]
        m_combined = torch.cat([m_feats, m_t_expanded], dim=-1)
        
        return self.net(m_combined).squeeze(-1) # [B, 48]

class BodyBlock(nn.Module):
    """
    1D Residual CNN over 24 body segments.
    """
    def __init__(self, in_dim, muscle_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim + muscle_feat_dim, hidden_dim)
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, b_t, m_feats):
        # b_t: [B, 24, 3] (sin, cos, dphi)
        # m_feats: [B, 48, D] -> aggregate to [B, 24, D]
        m24 = (m_feats[:, ::2] + m_feats[:, 1::2]) / 2.0
        
        x = torch.cat([b_t, m24], dim=-1)
        h = self.in_proj(x)
        
        # Conv expect [B, C, L]
        h_conv = h.transpose(1, 2)
        h_conv = F.relu(self.conv1(h_conv))
        h_conv = self.conv2(h_conv)
        
        h = h + h_conv.transpose(1, 2)
        h = F.relu(self.ln(h))
        
        return self.out_head(h)

class GlobalFeedbackBlock(nn.Module):
    """
    Predict COM velocity and generate neural feedback.
    """
    def __init__(self, body_hidden_dim, neural_hidden_dim):
        super().__init__()
        self.com_net = nn.Sequential(
            nn.Linear(body_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.feedback_net = nn.Linear(body_hidden_dim * 2, neural_hidden_dim)

    def forward(self, h_body):
        # h_body: [B, 24, D]
        # Pooling
        h_pool = torch.cat([h_body.mean(dim=1), h_body.max(dim=1)[0]], dim=-1) # [B, 2*D]
        
        d_com = self.com_net(h_pool)
        feedback = self.feedback_net(h_pool).unsqueeze(1) # [B, 1, D] for broadcasting to [B, 279, D]
        
        return d_com, feedback

class ModularSurrogate(nn.Module):
    def __init__(self, adj_gap, adj_syn, muscle_map, delta_scale=0.1):
        super().__init__()
        self.delta_scale = nn.Parameter(torch.tensor(delta_scale))
        
        self.neural_block = NeuralGNOBlock(in_dim=3, hidden_dim=64, out_dim=2, adj_gap=adj_gap, adj_syn=adj_syn)
        self.muscle_block = MuscleBlock(neural_hidden_dim=64, muscle_dim=48, muscle_map=muscle_map)
        self.body_block = BodyBlock(in_dim=3, muscle_feat_dim=64, hidden_dim=32, out_dim=3)
        self.global_block = GlobalFeedbackBlock(body_hidden_dim=32, neural_hidden_dim=64)
        
        # Map neural embeddings for muscle block reuse
        self.m_proj = (muscle_map[::2] + muscle_map[1::2]) / 2.0
        self.m_proj = torch.from_numpy(self.m_proj).float()

    def forward(self, z_t, u_t):
        """
        z_t: dict of tensors
        u_t: [B, 279]
        """
        # 1. Neural Update
        # Prepare node features: [v, s, input]
        n_feat = torch.stack([z_t['neural_v'], z_t['neural_s'], u_t], dim=-1)
        
        # We need a hidden state for the neural block to use for muscles
        # Let's run a projection first
        h_n = self.neural_block.node_init(n_feat)
        
        # Neural delta
        d_n = self.neural_block(n_feat) # [B, 279, 2]
        nv_next = z_t['neural_v'] + self.delta_scale * d_n[..., 0]
        ns_next = z_t['neural_s'] + self.delta_scale * d_n[..., 1]
        
        # 2. Muscle Update
        d_m = self.muscle_block(h_n, z_t['muscle'])
        m_next = z_t['muscle'] + self.delta_scale * d_m
        
        # 3. Body Update
        b_feat = torch.stack([z_t['phi_sin'], z_t['phi_cos'], z_t['dphi']], dim=-1)
        m_proj_feats = torch.matmul(self.m_proj.to(h_n.device), h_n) # [B, 48, D]
        
        d_b = self.body_block(b_feat, m_proj_feats) # [B, 24, 3]
        ps_next = z_t['phi_sin'] + self.delta_scale * d_b[..., 0]
        pc_next = z_t['phi_cos'] + self.delta_scale * d_b[..., 1]
        dp_next = z_t['dphi'] + self.delta_scale * d_b[..., 2]
        
        # Renormalize phi
        norm = torch.sqrt(ps_next**2 + pc_next**2 + 1e-8)
        ps_next = ps_next / norm
        pc_next = pc_next / norm
        
        # 4. Global Update
        h_b = self.body_block.in_proj(torch.cat([b_feat, (m_proj_feats[:, ::2] + m_proj_feats[:, 1::2])/2.0], dim=-1))
        # Wait, let's just get hidden from body block forward if we were cleaner.
        # For now, approximate or re-run a bit.
        
        d_com, feedback = self.global_block(h_b)
        cv_next = z_t['com_velocity'] + self.delta_scale * d_com
        
        return {
            'neural_v': nv_next,
            'neural_s': ns_next,
            'muscle': m_next,
            'phi_sin': ps_next,
            'phi_cos': pc_next,
            'dphi': dp_next,
            'com_velocity': cv_next
        }

# --- 2. Data Loading ---

class ModWormDataset(Dataset):
    def __init__(self, zarr_path, rollout_indices, window=32):
        root = zarr.open(zarr_path, mode='r')
        self.indices = rollout_indices
        self.window = window
        
        # Load EVERYTHING into memory
        self.data = {}
        keys = ['neural_v', 'neural_s', 'muscle', 'phi_sin', 'phi_cos', 'dphi', 'com_velocity', 'input']
        print(f"Pre-loading {len(rollout_indices)} rollouts into memory...")
        for k in keys:
            # We only need the rollouts in rollout_indices
            self.data[f't_{k}'] = torch.from_numpy(root[f'state_t/{k}'][rollout_indices]).float()
            if k != 'input':
                self.data[f'tp1_{k}'] = torch.from_numpy(root[f'state_tp1/{k}'][rollout_indices]).float()
        
        self.N_rollouts = len(rollout_indices)
        self.T = self.data['t_neural_v'].shape[1]
        
    def __len__(self):
        return self.N_rollouts * (self.T - self.window)

    def __getitem__(self, idx):
        r_local_idx = idx // (self.T - self.window)
        t_start = idx % (self.T - self.window)
        t_end = t_start + self.window
        
        item = {}
        for k in self.data.keys():
            item[k] = self.data[k][r_local_idx, t_start:t_end]
        return item

# --- 3. Training Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="outputs/modular_gno")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Connectome & Muscle Map
    # We find them relative to modWorm package or via saved metadata
    repo_root = Path.cwd()
    gap_path = repo_root / "modWorm/modWorm/data/conn_gap_adjust_Varshney.npy"
    syn_path = repo_root / "modWorm/modWorm/data/conn_syn_adjust_Varshney.npy"
    muscle_path = repo_root / "modWorm/modWorm/muscle_maps/muscle_map_adjust.npy"
    
    adj_gap = np.load(gap_path)
    adj_syn = np.load(syn_path)
    muscle_map = np.load(muscle_path)

    # Normalize adjacency
    def norm_adj(A):
        D = np.sum(A, axis=1)
        D_inv = 1.0 / (D + 1e-8)
        return A * D_inv[:, None]
        
    adj_gap = norm_adj(adj_gap)
    adj_syn = norm_adj(adj_syn)

    # Dataset
    root = zarr.open(args.data, mode='r')
    N_total = root['state_t/neural_v'].shape[0]
    indices = np.arange(N_total)
    np.random.shuffle(indices)
    train_idx = indices[:int(0.75*N_total)]
    test_idx = indices[int(0.75*N_total):]
    
    train_ds = ModWormDataset(args.data, train_idx, window=args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ModularSurrogate(adj_gap, adj_syn, muscle_map).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    weights = {
        'neural_v': 1.0, 'neural_s': 1.0, 'muscle': 1.0,
        'phi_sin': 2.0, 'phi_cos': 2.0, 'dphi': 2.0,
        'com_velocity': 3.0
    }

    history = {'train_loss': [], 'test_rollout_loss': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for b_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Init state from batch
            curr_z = {k: batch[f't_{k}'].to(device) for k in weights.keys()}
            u_seq = batch['t_input'].to(device)
            
            # Rollout
            loss = 0
            for t in range(args.window):
                # Get t-th step
                z_t = {k: v[:, t] for k, v in curr_z.items()}
                u_t = u_seq[:, t]
                
                z_next_pred = model(z_t, u_t)
                
                # Step loss
                step_loss = 0
                for k, w in weights.items():
                    target = batch[f'tp1_{k}'][:, t].to(device)
                    step_loss += w * criterion(z_next_pred[k], target)
                
                loss += step_loss
                # Mixed rollout: first t steps are ground truth, then predicted?
                # Actually, standard RNN rollout:
                if t < args.window - 1:
                    # Update curr_z for next step with predicted values to do closed-loop
                    # This is more robust for surrogates.
                    for k in weights.keys():
                        # Use a mix of pred and target (Scheduled Sampling)
                        # For now, 50% chance of pred
                        if np.random.random() < 0.5:
                            curr_z[k][:, t+1] = z_next_pred[k].detach()

            
            loss = loss / args.window
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if b_idx % 50 == 0:
                print(f"Epoch {epoch:03d} | Batch {b_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
            
        avg_train = total_loss / len(train_loader)
        history['train_loss'].append(avg_train)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train:.6f}")
        
        # Eval rollout on one test sample
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                t_idx = test_idx[0]
                # Full rollout T-1 steps
                T_eval = root['state_t/neural_v'].shape[1]
                z_t = {k: torch.from_numpy(root[f'state_t/{k}'][t_idx, 0:1]).float().to(device) for k in weights.keys()}
                u_seq = torch.from_numpy(root['state_t/input'][t_idx]).float().to(device)
                
                test_loss = 0
                for t in range(T_eval):
                    z_pred = model(z_t, u_seq[t:t+1])
                    for k, w in weights.items():
                        target = torch.from_numpy(root[f'state_tp1/{k}'][t_idx, t:t+1]).float().to(device)
                        test_loss += w * criterion(z_pred[k], target).item()
                    z_t = z_pred
                
                history['test_rollout_loss'].append(test_loss)
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train:.6f} | Test Rollout Loss: {test_loss:.6f}")

    # --- 4. Plotting & Metrics ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.yscale('log')
    plt.legend()
    plt.savefig(outdir / "loss_curves.png")
    
    # Save Model
    torch.save(model.state_dict(), outdir / "model.pt")
    
    # Differentiability Check
    model.eval()
    test_idx_0 = test_idx[0]
    z0 = {k: torch.from_numpy(root[f'state_t/{k}'][test_idx_0, 0:1]).float().to(device) for k in weights.keys()}
    u0 = torch.from_numpy(root['state_t/input'][test_idx_0, 0:1]).float().to(device)
    u0.requires_grad_(True)
    
    z1 = model(z0, u0)
    com_speed = torch.norm(z1['com_velocity'])
    com_speed.backward()
    grad_norm = u0.grad.norm().item()
    print(f"\nDifferentiability check: grad norm wrt input = {grad_norm:.6f}")

    # Horizon Metrics
    horizons = [1, 4, 8, 16, 32]
    metrics = {}
    model.eval()
    with torch.no_grad():
        for h in horizons:
            errs = []
            for t_idx in test_idx[:5]: # Take 5 test rollouts
                T_max = root['state_t/neural_v'].shape[1] - h
                for t_start in range(0, T_max, 20):
                    z_curr = {k: torch.from_numpy(root[f'state_t/{k}'][t_idx, t_start:t_start+1]).float().to(device) for k in weights.keys()}
                    u_seq = torch.from_numpy(root['state_t/input'][t_idx, t_start:t_start+h]).float().to(device)
                    
                    for step in range(h):
                        z_curr = model(z_curr, u_seq[step:step+1])
                    
                    # Compute relative L2 error for com_velocity as proxy
                    target = torch.from_numpy(root['state_tp1/com_velocity'][t_idx, t_start+h-1:t_start+h]).float().to(device)
                    num = torch.norm(z_curr['com_velocity'] - target)
                    den = torch.norm(target) + 1e-6
                    errs.append((num/den).item())
            metrics[str(h)] = np.median(errs)

    with open(outdir / "final_metrics.json", "w") as f:
        json.dump({"horizon_metrics": metrics, "grad_norm": grad_norm}, f, indent=2)

    # --- Phi Heatmap Plot ---
    t_idx = test_idx[0]
    T_eval = root['state_t/phi_sin'].shape[1]
    z_t = {k: torch.from_numpy(root[f'state_t/{k}'][t_idx, 0:1]).float().to(device) for k in weights.keys()}
    u_seq = torch.from_numpy(root['state_t/input'][t_idx]).float().to(device)
    
    phi_preds = []
    with torch.no_grad():
        for t in range(T_eval):
            z_pred = model(z_t, u_seq[t:t+1])
            # Reconstruction phi from sin/cos is not needed if we just plot sin
            phi_preds.append(z_pred['phi_sin'][0].cpu().numpy())
            z_t = z_pred
            
    phi_preds = np.stack(phi_preds)
    phi_gt = root['state_t/phi_sin'][t_idx]
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    im0 = ax[0].imshow(phi_gt.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax[0].set_title("Ground Truth (Phi Sin)")
    im1 = ax[1].imshow(phi_preds.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax[1].set_title("Modular GNO Prediction")
    plt.tight_layout()
    plt.savefig(outdir / "test_rollout_phi_heatmap.png")

    print("\nDONE.")

if __name__ == "__main__":
    main()
