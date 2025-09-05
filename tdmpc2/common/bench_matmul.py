#!/usr/bin/env python3
"""
Benchmark batch matmul vs loop vs vmap vs torch.distributed (CPU).
Save as bench_matmul.py and run with Python (single-process) or torchrun (distributed).

Examples:
  Single-process all modes:
    python bench_matmul.py --mode all --pop 1024 --M 128 --N 128 --K 128 --reps 20 --warmup 5 --threads 4

  Distributed (4 processes on one machine):
    torchrun --nproc_per_node=4 bench_matmul.py --mode dist --pop 4096 --M 128 --N 128 --K 128 --reps 20 --warmup 5

Notes:
  - vmap: tries torch.func.vmap, then functorch.vmap. If not available it will skip the vmap benchmark.
  - Distributed mode expects torchrun/torch.distributed env (RANK, WORLD_SIZE, LOCAL_RANK). Uses Gloo backend.
  - You can limit CPU threads via --threads to get reproducible comparisons.
"""

import argparse
import time
import math
import os
import sys

import torch
import torch.multiprocessing as mp

try:
    import torch.distributed as dist
except Exception:
    dist = None

# try vmap from torch.func or functorch
_vmap = None
try:
    from torch.func import vmap as _torch_vmap  # PyTorch 2.x
    _vmap = _torch_vmap
except Exception:
    try:
        from functorch import vmap as _ft_vmap
        _vmap = _ft_vmap
    except Exception:
        _vmap = None


def now():
    return time.perf_counter()


def bench_func(func, warmup, reps, *args, **kwargs):
    # warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    # timed runs
    t0 = now()
    for _ in range(reps):
        func(*args, **kwargs)
    t1 = now()
    total = t1 - t0
    avg = total / reps
    return total, avg


def batch_mm(a, b):
    # torch.bmm for 3D batches
    return torch.bmm(a, b)


def loop_mm(a, b):
    # naive Python loop
    out = []
    for i in range(a.shape[0]):
        out.append(a[i].matmul(b[i]))
    return torch.stack(out, dim=0)


def vmap_mm(a, b):
    if _vmap is None:
        raise RuntimeError("vmap not available")
    fn = lambda x, y: x.matmul(y)
    return _vmap(fn)(a, b)


def distributed_worker(rank, world_size, args):
    # This function is not used for torchrun; implemented only if you want mp.spawn approach.
    raise NotImplementedError("Use torchrun to run distributed mode (see README).")


def run_distributed(a, b, warmup, reps):
    """
    Assumes the process is launched by torchrun (so RANK, WORLD_SIZE env vars exist).
    Each rank computes a slice of the batch and we compute timing for the slowest rank (max).
    """
    if dist is None:
        raise RuntimeError("torch.distributed not available in this build of PyTorch.")

    # Initialize process group using environment variables from torchrun
    # Backend gloo for CPU
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total = a.shape[0]

    # split indices fairly
    per = (total + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, total)
    local_count = max(0, end - start)

    # slice local tensors
    a_local = a[start:end].contiguous()
    b_local = b[start:end].contiguous()

    # ensure same data / memory layout
    torch.set_num_threads(1)  # avoid oversubscription inside each rank

    # warmup
    for _ in range(warmup):
        if local_count > 0:
            _ = torch.bmm(a_local, b_local)

    # time local
    t0 = now()
    for _ in range(reps):
        if local_count > 0:
            _ = torch.bmm(a_local, b_local)
    t1 = now()
    local_total = t1 - t0

    # compute the maximum time among ranks (worst-case)
    local_tensor = torch.tensor([local_total], dtype=torch.float64)
    dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    max_time = local_tensor.item()

    # optionally print from rank 0
    if rank == 0:
        print(f"[distributed] world_size={world_size} total_batches={total} per_rank≈{per}")
        print(f"[distributed] slowest rank total time (all reps): {max_time:.6f} s")
        avg = max_time / reps
        print(f"[distributed] avg per rep (worst rank): {avg:.6f} s")
    # finalize
    dist.barrier()
    dist.destroy_process_group()


def flops_per_matmul(M, N, K):
    # approximate 2*M*N*K FLOPs per matmul
    return 2.0 * M * N * K


def humanize_flops(flops_per_sec):
    # return approximate GFLOPS or MFLOPS
    if flops_per_sec >= 1e9:
        return f"{flops_per_sec/1e9:.2f} GFLOPS"
    if flops_per_sec >= 1e6:
        return f"{flops_per_sec/1e6:.2f} MFLOPS"
    return f"{flops_per_sec:.2f} FLOPS"


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch matmul vs loop vs vmap vs distributed (CPU)")
    parser.add_argument("--mode", choices=["batch", "loop", "vmap", "dist", "all"], default="all")
    parser.add_argument("--pop", type=int, default=1024, help="POP_SIZE (batch count)")
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="torch.set_num_threads() (0 = don't set)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # set threads
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    # device: CPU only
    device = torch.device("cpu")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)

    POP = args.pop
    M = args.M
    N = args.N
    K = args.K

    print(f"Running on device={device} dtype={dtype} POP={POP} M={M} N={N} K={K}")
    if args.threads:
        print(f"torch.set_num_threads({args.threads})")

    # Create tensors
    a = torch.randn(POP, M, N, device=device, dtype=dtype)
    b = torch.randn(POP, N, K, device=device, dtype=dtype)

    # quick sanity (one run)
    out = torch.bmm(a[:1], b[:1])
    assert out.shape == (1, M, K)

    total_flops_single = flops_per_matmul(M, N, K) * POP

    modes = [args.mode] if args.mode != "all" else ["batch", "loop", "vmap", "dist"]

    for mode in modes:
        print("=" * 70)
        print(f"MODE: {mode}")

        if mode == "batch":
            total, avg = bench_func(lambda: batch_mm(a, b), args.warmup, args.reps)
            print(f"[batch] total {total:.6f} s, avg {avg:.6f} s")
            gflops = total_flops_single / avg
            print(f"[batch] throughput ~ {humanize_flops(gflops)} (approx)")

        elif mode == "loop":
            total, avg = bench_func(lambda: loop_mm(a, b), args.warmup, args.reps)
            print(f"[loop] total {total:.6f} s, avg {avg:.6f} s")
            gflops = total_flops_single / avg
            print(f"[loop] throughput ~ {humanize_flops(gflops)} (approx)")

        elif mode == "vmap":
            if _vmap is None:
                print("[vmap] vmap not available in this Python environment; skipping.")
                continue
            # vmap may allocate; do it via function
            total, avg = bench_func(lambda: vmap_mm(a, b), args.warmup, args.reps)
            print(f"[vmap] total {total:.6f} s, avg {avg:.6f} s")
            gflops = total_flops_single / avg
            print(f"[vmap] throughput ~ {humanize_flops(gflops)} (approx)")

        elif mode == "dist":
            # distributed mode must be launched via torchrun which sets RANK/WORLD_SIZE env vars
            required_env = ("RANK" in os.environ) or ("OMPI_COMM_WORLD_RANK" in os.environ) or ("LOCAL_RANK" in os.environ)
            if not required_env:
                print("[dist] Warning: distributed mode expects torchrun or env vars (RANK/WORLD_SIZE).")
                print("         Example: torchrun --nproc_per_node=4 bench_matmul.py --mode dist ...")
                # fallback: run a local single-process simulation of distributed timing (just do local bmm)
                print("[dist] Running local bmm timing as fallback (not distributed).")
                total, avg = bench_func(lambda: batch_mm(a, b), args.warmup, args.reps)
                print(f"[dist:fallback] total {total:.6f} s, avg {avg:.6f} s")
                continue

            # run distributed routine
            run_distributed(a, b, args.warmup, args.reps)

        else:
            print(f"Unknown mode: {mode}")

    print("Done.")


if __name__ == "__main__":
    main()

