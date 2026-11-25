"""train_head.py with the trainer's GPU memory taken up front (written 2026-11-24 for the open-loop memory diagnosis).

The frozen trainer is imported and run unchanged. Before it starts, this process allocates one block of
TRAINER_RESERVE_GB GiB plus 512 one-MiB blocks and frees them. PyTorch's caching allocator does not hand freed blocks
back to the device, so everything the trainer allocates afterwards is carved out of that cache and no longer depends
on device memory that the serving process may have filled in the meantime. The training computation is unchanged."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
G = float(os.environ.get("TRAINER_RESERVE_GB", "0"))
if G > 0:
    big = torch.empty(int(G * 2 ** 30), dtype=torch.uint8, device="cuda")        # large pool: one block, split on demand
    small = [torch.empty(2 ** 20, dtype=torch.uint8, device="cuda") for _ in range(512)]   # small pool: 2 MiB segments
    del big, small
    torch.cuda.reset_peak_memory_stats()                                           # peak_mem_gb in the log stays the trainer's own
    print(f"RESERVED {torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB (allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f})",
          flush=True)

import train_head  # noqa: E402  (frozen trainer)

if __name__ == "__main__":
    train_head.main()
