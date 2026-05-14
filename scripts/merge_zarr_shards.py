import argparse
from pathlib import Path
import json
import zarr
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing shard .zarr folders")
    parser.add_argument("--output", type=str, required=True, help="Output merged .zarr path")
    parser.add_argument("--metadata", type=str, required=True, help="Output merged .json metadata path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_zarr_path = Path(args.output)
    out_json_path = Path(args.metadata)

    shards = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.suffix == '.zarr'])
    if not shards:
        print("No shards found.")
        return

    print(f"Found {len(shards)} shards to merge.")

    # Read the first shard to get structure
    first_root = zarr.open(str(shards[0]), mode='r')
    
    # Create the output zarr store
    out_root = zarr.open(str(out_zarr_path), mode='w')

    # Initialize datasets using the shapes from the first shard, but with N dimension open/extendable
    # Zarr handles append natively
    
    total_N = 0
    all_metadata = []

    for shard_idx, shard_path in enumerate(shards):
        print(f"Merging {shard_path}...")
        shard_root = zarr.open(str(shard_path), mode='r')
        
        # Read JSON metadata
        json_path = shard_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
                if isinstance(meta, list):
                    all_metadata.extend(meta)
                else:
                    all_metadata.append(meta)

        def append_recursive(in_group, out_group):
            for key, item in in_group.items():
                if isinstance(item, zarr.hierarchy.Group):
                    if key not in out_group:
                        out_group.create_group(key)
                    append_recursive(item, out_group[key])
                elif isinstance(item, zarr.core.Array):
                    if key not in out_group:
                        # Create array with chunks from first shard
                        shape = list(item.shape)
                        shape[0] = 0 # Empty initially
                        out_group.create_dataset(
                            key,
                            shape=tuple(shape),
                            chunks=item.chunks,
                            dtype=item.dtype
                        )
                    out_group[key].append(item[:], axis=0)

        append_recursive(shard_root, out_root)
        total_N += shard_root['input'].shape[0] if 'input' in shard_root else shard_root['state_t/input'].shape[0]

    print(f"Merged {len(shards)} shards. Total N = {total_N}.")

    with open(out_json_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved metadata to {out_json_path}")

if __name__ == "__main__":
    main()
