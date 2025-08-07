
import time
import functools
import numpy as np
import torch

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
            print(f"Function [{func.__name__}] Device [{device}] avg time over {runs} runs: {avg_time:.3f} \pm {std_time:.3f} ms")
            return result

        return wrapper
    return decorator