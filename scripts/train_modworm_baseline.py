#!/usr/bin/env python3
"""
Train a differentiable surrogate baseline for modWorm rollouts.

Usage:
  source .venv/bin/activate
  python scripts/train_modworm_baseline.py --epochs 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, depth=4, delta_scale=0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.delta_scale = delta_scale

    def forward(self, z, u):
        x = torch.cat([z, u], dim=-1)
        delta = self.mlp(x)
        return z + self.delta_scale * delta


class ResidualGRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, delta_scale=0.1):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.delta_scale = delta_scale
        self.hidden_dim = hidden_dim

    def forward(self, z, u, h=None):
        x = torch.cat([z, u], dim=-1)
        h_next = self.gru(x, h)
        delta = self.head(h_next)
        return z + self.delta_scale * delta, h_next


def load_data(zarr_path, stats_path):
    root = zarr.open(str(zarr_path), mode='r')
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    train_indices = stats['train_indices']
    test_indices = stats['test_indices']

    # We use state_t and state_tp1 groups
    def get_group(group_name):
        g = root[group_name]
        data = {
            "v": np.array(g['neural_v']),
            "s": np.array(g['neural_s']),
            "muscle": np.array(g['muscle']),
            "phi_sin": np.array(g['phi_sin']),
            "phi_cos": np.array(g['phi_cos']),
            "dphi": np.array(g['dphi']),
            "com_vel": np.array(g['com_velocity']),
            "input": np.array(g['input'])
        }
        return data

    st = get_group('state_t')
    stp1 = get_group('state_tp1')

    # Construct z_t
    def concat_z(d):
        return np.concatenate([
            d['v'], d['s'], d['muscle'], d['phi_sin'], d['phi_cos'], d['dphi'], d['com_vel']
        ], axis=-1)

    z_t = concat_z(st)
    u_t = st['input']
    z_tp1 = concat_z(stp1)

    return z_t, u_t, z_tp1, train_indices, test_indices, stats


def weighted_mse_loss(pred, target):
    # Dims: v(279), s(279), m(48), ps(24), pc(24), dp(24), cv(2)
    # Total: 680
    v_dim, s_dim, m_dim, p_dim, c_dim, d_dim, cv_dim = 279, 279, 48, 24, 24, 24, 2
    
    idx = 0
    l_v = nn.MSELoss()(pred[..., idx:idx+v_dim], target[..., idx:idx+v_dim])
    idx += v_dim
    l_s = nn.MSELoss()(pred[..., idx:idx+s_dim], target[..., idx:idx+s_dim])
    idx += s_dim
    l_m = nn.MSELoss()(pred[..., idx:idx+m_dim], target[..., idx:idx+m_dim])
    idx += m_dim
    l_ps = nn.MSELoss()(pred[..., idx:idx+p_dim], target[..., idx:idx+p_dim])
    idx += p_dim
    l_pc = nn.MSELoss()(pred[..., idx:idx+c_dim], target[..., idx:idx+c_dim])
    idx += c_dim
    l_dp = nn.MSELoss()(pred[..., idx:idx+d_dim], target[..., idx:idx+d_dim])
    idx += d_dim
    l_cv = nn.MSELoss()(pred[..., idx:idx+cv_dim], target[..., idx:idx+cv_dim])
    
    loss = 1.0 * l_v + 1.0 * l_s + 1.0 * l_m + 2.0 * l_ps + 2.0 * l_pc + 2.0 * l_dp + 3.0 * l_cv
    return loss, {
        "l_v": l_v.item(), "l_s": l_s.item(), "l_m": l_m.item(),
        "l_ps": l_ps.item(), "l_pc": l_pc.item(), "l_dp": l_dp.item(), "l_cv": l_cv.item()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--outdir", type=str, default="outputs/baseline_surrogate")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_t_raw, u_t_raw, z_tp1_raw, train_idx, test_idx, stats = load_data(args.data, args.stats)
    
    N, T_minus_1, Z_dim = z_t_raw.shape
    U_dim = u_t_raw.shape[-1]
    print(f"Loaded data: N={N}, T-1={T_minus_1}, Z_dim={Z_dim}, U_dim={U_dim}")
    print(f"Train indices: {train_idx}, Test indices: {test_idx}")

    z_t = torch.from_numpy(z_t_raw).float()
    u_t = torch.from_numpy(u_t_raw).float()
    z_tp1 = torch.from_numpy(z_tp1_raw).float()

    if args.model == "mlp":
        model = ResidualMLP(Z_dim + U_dim, Z_dim).to(device)
    else:
        model = ResidualGRU(Z_dim + U_dim, Z_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    train_log = []
    
    for epoch in range(args.epochs):
        model.train()
        # Random sampling of windows
        batch_rollout_idx = np.random.choice(train_idx, size=args.batch_size)
        batch_t0 = np.random.randint(0, T_minus_1 - args.window, size=args.batch_size)
        
        loss_total = 0
        optimizer.zero_grad()
        
        # We process one batch by unrolling
        # (batch, Z_dim)
        z_curr = z_t[batch_rollout_idx, batch_t0].to(device)
        h = None
        
        rollout_losses = []
        for k in range(args.window):
            u_curr = u_t[batch_rollout_idx, batch_t0 + k].to(device)
            z_target = z_tp1[batch_rollout_idx, batch_t0 + k].to(device)
            
            if args.model == "mlp":
                z_next = model(z_curr, u_curr)
            else:
                z_next, h = model(z_curr, u_curr, h)
            
            step_loss, _ = weighted_mse_loss(z_next, z_target)
            rollout_losses.append(step_loss)
            z_curr = z_next # Rollout
            
        loss = torch.stack(rollout_losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Eval on test set
            model.eval()
            with torch.no_grad():
                # Simple one-step and rollout eval on first test sample
                tid = test_idx[0]
                z_test = z_t[tid].to(device)
                u_test = u_t[tid].to(device)
                z_target_test = z_tp1[tid].to(device)
                
                # Full test rollout
                z_unroll = z_test[0:1] # (1, Z_dim)
                h_test = None
                test_rollout_preds = []
                for k in range(min(128, T_minus_1)):
                    ut = u_test[k:k+1]
                    if args.model == "mlp":
                        z_unroll = model(z_unroll, ut)
                    else:
                        z_unroll, h_test = model(z_unroll, ut, h_test)
                    test_rollout_preds.append(z_unroll.squeeze(0))
                
                test_rollout_preds = torch.stack(test_rollout_preds)
                test_loss, test_breakdown = weighted_mse_loss(test_rollout_preds, z_target_test[:len(test_rollout_preds)])
                
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.6f} | Test Rollout Loss: {test_loss.item():.6f}")
            train_log.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                **test_breakdown
            })

    # Save
    torch.save(model.state_dict(), outdir / "model.pt")
    with open(outdir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    # Plots
    epochs = [l['epoch'] for l in train_log]
    train_losses = [l['train_loss'] for l in train_log]
    test_losses = [l['test_loss'] for l in train_log]

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Rollout Loss')
    plt.plot(epochs, test_losses, label='Test Rollout Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outdir / "loss_curves.png")

    # Final eval for horizon errors
    model.eval()
    horizons = [1, 4, 8, 16, 32]
    horizon_metrics = {}
    
    with torch.no_grad():
        for h in horizons:
            errs = []
            for tid in test_idx:
                z_in = z_t[tid].to(device)
                u_in = u_t[tid].to(device)
                z_tar = z_tp1[tid].to(device)
                
                # Many windows of size h
                for t0 in range(0, T_minus_1 - h, h):
                    zc = z_in[t0:t0+1]
                    hc = None
                    for k in range(h):
                        if args.model == "mlp":
                            zc = model(zc, u_in[t0+k:t0+k+1])
                        else:
                            zc, hc = model(zc, u_in[t0+k:t0+k+1], hc)
                    rel_err = torch.norm(zc - z_tar[t0+h-1:t0+h]) / (torch.norm(z_tar[t0+h-1:t0+h]) + 1e-6)
                    errs.append(rel_err.item())
            horizon_metrics[h] = np.median(errs)

    plt.figure()
    plt.plot(horizons, [horizon_metrics[h] for h in horizons], marker='o')
    plt.xlabel('Horizon')
    plt.ylabel('Median Relative L2 Error')
    plt.savefig(outdir / "horizon_errors.png")
    
    with open(outdir / "final_metrics.json", "w") as f:
        json.dump({"horizon_metrics": horizon_metrics}, f, indent=2)

    # Rollout Visualization
    with torch.no_grad():
        tid = test_idx[0]
        z_unroll = z_t[tid, 0:1].to(device)
        u_test = u_t[tid].to(device)
        preds = []
        h_test = None
        for k in range(min(128, T_minus_1)):
            if args.model == "mlp":
                z_unroll = model(z_unroll, u_test[k:k+1])
            else:
                z_unroll, h_test = model(z_unroll, u_test[k:k+1], h_test)
            preds.append(z_unroll.squeeze(0).cpu().numpy())
        preds = np.stack(preds)
        targets = z_tp1_raw[tid, :len(preds)]

        # Phi is indices [279+279+48 : 279+279+48+24+24] -> [606:654]
        # sin(phi) is [606:630], cos(phi) is [630:654]
        phi_sin_pred = preds[:, 606:630]
        phi_cos_pred = preds[:, 630:654]
        phi_pred = np.arctan2(phi_sin_pred, phi_cos_pred)
        
        phi_sin_tar = targets[:, 606:630]
        phi_cos_tar = targets[:, 630:654]
        phi_tar = np.arctan2(phi_sin_tar, phi_cos_tar)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(phi_tar.T, aspect='auto', cmap='RdBu', origin='lower')
        axes[0].set_title('True Phi')
        axes[1].imshow(phi_pred.T, aspect='auto', cmap='RdBu', origin='lower')
        axes[1].set_title('Pred Phi')
        axes[2].imshow(np.abs(phi_pred - phi_tar).T, aspect='auto', cmap='viridis', origin='lower')
        axes[2].set_title('Abs Error')
        plt.savefig(outdir / "test_rollout_phi_heatmap.png")

        # COM Vel is [678:680]
        cv_pred = preds[:, 678:680]
        cv_tar = targets[:, 678:680]
        plt.figure()
        plt.plot(cv_tar[:, 0], label='True VX')
        plt.plot(cv_pred[:, 0], '--', label='Pred VX')
        plt.plot(cv_tar[:, 1], label='True VY')
        plt.plot(cv_pred[:, 1], '--', label='Pred VY')
        plt.legend()
        plt.savefig(outdir / "test_rollout_com_velocity.png")

    # Gradient Check
    model.eval()
    tid = test_idx[0]
    u_grad = u_t[tid, :32].to(device).clone().detach().requires_grad_(True)
    z0 = z_t[tid, 0:1].to(device)
    zc = z0
    hc = None
    for k in range(32):
        if args.model == "mlp":
            zc = model(zc, u_grad[k:k+1])
        else:
            zc, hc = model(zc, u_grad[k:k+1], hc)
    
    # Target: mean predicted com_velocity
    target_metric = zc[:, 678:680].mean()
    target_metric.backward()
    grad_norm = u_grad.grad.norm().item()
    print(f"\nDifferentiability check: grad norm wrt input = {grad_norm:.6f}")

    print("\nDONE.")

if __name__ == "__main__":
    main()
