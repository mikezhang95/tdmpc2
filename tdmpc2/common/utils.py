
import time
import functools
import numpy as np
import torch
import random

is_cuda = torch.cuda.is_available()

def benchmark_torch_function(runs=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            execution_times = []
            result = None
            for _ in range(runs):
                if is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    result = func(*args, **kwargs)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)  # ms
                else:
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    elapsed_time = (end_time - start_time) * 1000  # ms

                execution_times.append(elapsed_time)

            execution_times = np.array(execution_times)
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            device = "GPU" if is_cuda else "CPU"
            print(f"Function [{func.__name__}] Device [{device}] avg time over {runs} runs: {avg_time:.2f} \pm {std_time:.2f} ms")
            return result

        return wrapper
    return decorator

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def np_to_torch(t, device='cpu'):
    if t is None:
        return None
    else:
        return torch.tensor(t).to(device).float()

def torch_to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def sample_negatives(x, y, k):
    """
    x: [B, 1] tensor
    return: [B, k] tensor, where each row = [x_i, (k-1) negatives]
    """
    B = x.size(0)
    device = x.device

    # Expand original positives
    positives_x = x.view(B, 1)
    positives_y = y.view(B, 1)

    rand_idx = torch.randint(0, B-1, (B, k-1), device=device)
    mask = rand_idx >= torch.arange(B, device=device).unsqueeze(1)
    rand_idx += mask.long()  

    negatives_x = x[rand_idx].squeeze(-1)  # shape [B, k-1]
    negatives_y = y[rand_idx].squeeze(-1)  # shape [B, k-1]

    out_x = torch.cat([positives_x, negatives_x], dim=1)
    out_y = torch.cat([positives_y, negatives_y], dim=1)
    return out_x, out_y
