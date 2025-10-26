
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

import control 
def dlqr_with_linear_terms(A, B, Q, R, q=None, r=None, tol=1e-12):
    """
    Solve discrete-time infinite-horizon LQR with linear terms:
      J = sum_t x'Qx + u'Ru + 2 q'x + 2 r' u

    Returns:
      K: [n_u, n_x]   (state feedback)
      k0: [n_u,]      (constant offset, so u = -K x - k0)
      S: Riccati solution
      E: eigenvalues (closed-loop)
    """
    n_x = A.shape[0]
    n_u = B.shape[1]

    if q is None:
        q = np.zeros((n_x,))
    if r is None:
        r = np.zeros((n_u,))

    # 1) solve standard dlqr for S and K
    K, S, E = control.dlqr(A, B, Q, R)  # K has shape (n_u, n_x)
    # control.dlqr returns K as numpy array shape (n_u, n_x)

    # Ensure shapes
    q = q.reshape(-1)
    r = r.reshape(-1)

    # 2) compute M = R + B^T S B, should be positive definite
    M = R + B.T.dot(S).dot(B)

    # 3) solve for s: (I - (A - B K)^T) s = q + K^T r
    Acl = A - B.dot(K)  # closed-loop A
    I = np.eye(n_x)
    lhs = I - Acl.T
    rhs = q + K.T.dot(r)

    # Check invertibility / solve robustly
    # Use np.linalg.solve (or lstsq if singular)
    try:
        s = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # fallback to least-squares
        s, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    # 4) compute k0 by solving M k0 = B^T s + r
    rhs_k = B.T.dot(s) + r
    k0 = np.linalg.solve(M, rhs_k)

    return K, k0, S, E


# from mpc import mpc
# from mpc.mpc import QuadCost, LinDx
#     def act(self, z, refresh=False):

#         bs, n_state = z.shape
#         n_ctrl = self.action_dim
#         horizon = 10  # MPC horizon

#         A, B, Q, R = self.get_lqr_matrics(discount=1.0)

#         # Step 2. Construct block matrices for RePE
#         # Combine A, B into F = [A | B] (horizon, B, n_state, n_state + n_ctrl)
#         F = torch.cat([A, B], dim=1).unsqueeze(0).unsqueeze(0).repeat(horizon, bs, 1, 1)  # (horizon, bs, n_state, n_state + n_ctrl)

#         # Construct quadratic cost matrix C for (x,u)
#         C_block = torch.zeros(horizon, bs, n_state + n_ctrl, n_state + n_ctrl, device=z.device)
#         C_block[:, :, :n_state, :n_state] = Q.unsqueeze(0).unsqueeze(0).repeat(horizon, bs, 1, 1)
#         C_block[:, :, n_state:, n_state:] = R.unsqueeze(0).unsqueeze(0).repeat(horizon, bs, 1, 1)
#         c_block = torch.zeros((horizon, bs, n_state + n_ctrl), device=z.device)

#         # Step 3. Control bounds
#         u_lower = -torch.ones(horizon, bs, n_ctrl, device=z.device) * 1.0
#         u_upper = torch.ones(horizon, bs, n_ctrl, device=z.device) * 1.0

#         x_init = z  # current state as MPC initial condition
#         x_pred, u_pred, _ = mpc.MPC(
#             n_state=n_state,
#             n_ctrl=n_ctrl,
#             T=horizon,
#             u_lower=u_lower,
#             u_upper=u_upper,
#             lqr_iter=10,
#             backprop=False,
#             verbose=0,
#             exit_unconverged=False,
#         )(x_init, QuadCost(C_block, c_block), LinDx(F))
#         u_pred = u_pred[0] # Take the first action only

#         if self.cfg.dynamic_structure == 'companion_fixed':
#             u_pred = self._action_encoder.inverse(z, u_pred)
#         return u_pred.clamp(-1.0, 1.0)