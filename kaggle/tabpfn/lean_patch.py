# Memory-lean KV-cache build for TabPFN-3.5 (tabpfn==9.0.0).
# The stock fit_with_cache path runs every ICL layer over all ~560k context rows
# at once (full q/k/v + fp32 norm copies + unchunked MLP) -> OOM on a 16 GB T4.
# This patch computes K/V for the context in row chunks, then the attention
# output and the MLP in row chunks, updating the residual stream in place.
# Same math as the original (queries are independent given K/V; SSMax scaling
# depends only on the K length), so predictions are unchanged.
#
# v2 (10.10): two more savings, found from the OOM tracebacks at 653k / 783k context rows:
#  - K/V are kept in the (B, H, N, D) layout SDPA wants and passed to torch SDPA directly. The stock
#    wrapper permutes (B, N, H, D) and calls .contiguous(), i.e. copies the full 16-head K and V
#    (1.5 GiB each at 783k rows) on every query chunk.
#  - the many-class decoder keys (pre-head MLP, 2048 hidden, then the key projection) are projected in
#    row chunks instead of over all context rows at once (2.5 GiB GELU buffer at 653k rows).
# MATH is left out of the SDPA backends on CUDA so that a missing fast kernel fails at once instead of
# materialising the attention matrix (LEAN_BACKENDS=math adds it, for CPU tests).
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tabpfn.architectures import tabpfn_v3_5 as A

CH = int(os.environ.get("LEAN_CHUNK", 16384))
BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
if os.environ.get("LEAN_BACKENDS") == "math":
    BACKENDS.append(SDPBackend.MATH)
_calls = [0, None]
_orig_forward = A.ICLTransformerBlock.forward
_orig_decoder_keys = A.MultiTaskHeads.project_decoder_keys


def _attend(att, q_BSHD, K_BHND, V_BHND):
    """Attention of a query chunk over the context K/V; returns (B, S, H, D)."""
    if att.softmax_scaling_layer is not None:
        q_BSHD = att.softmax_scaling_layer(q_BSHD, K_BHND.shape[2])
    q = q_BSHD.permute(0, 2, 1, 3).to(K_BHND.dtype).contiguous()
    with sdpa_kernel(BACKENDS):
        out = F.scaled_dot_product_attention(q, K_BHND, V_BHND)
    return out.permute(0, 2, 1, 3)


def _lean_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor=None, *,
                  cached_kv=None, return_kv=False):
    if not return_kv or cached_kv is not None:
        return _orig_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor,
                             cached_kv=cached_kv, return_kv=return_kv)
    att = self.icl_attention
    x = x_BRE.contiguous()
    B, R, _ = x.shape
    N = R if single_eval_pos is None else single_eval_pos
    nh = att.num_kv_heads_test
    with torch.no_grad():
        K = V = None
        for s in range(0, N, CH):
            e = min(s + CH, N)
            h = self.layernorm(x[:, s:e])
            k = att.k_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
            v = att.v_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
            k = att.k_norm(k)
            if K is None:
                K = torch.empty((B, att.num_kv_heads, N, att.head_dim), dtype=v.dtype, device=x.device)
                V = torch.empty_like(K)
            K[:, :, s:e] = k.to(v.dtype).permute(0, 2, 1, 3)
            V[:, :, s:e] = v.permute(0, 2, 1, 3)
            del h, k, v
        starts = list(range(0, N, CH)) + list(range(N, R, CH))
        for s in starts:
            e = min(s + CH, N if s < N else R)
            xc = x[:, s:e]
            q = att.q_norm(att.q_projection(self.layernorm(xc)).view(B, e - s, att.num_heads, att.head_dim))
            if s < N or nh is None or N == R:
                out = _attend(att, q, K, V)
            else:
                out = _attend(att, q, K[:, :nh], V[:, :nh])
            xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))
            del q, out
        if nh is not None:
            k_cache = K[:, :nh].permute(0, 2, 1, 3).contiguous()
            v_cache = V[:, :nh].permute(0, 2, 1, 3).contiguous()
        else:
            k_cache, v_cache = K.permute(0, 2, 1, 3).contiguous(), V.permute(0, 2, 1, 3).contiguous()
        del K, V
        kv_entry = A.KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())
        for s in range(0, R, CH):
            xc = x[:, s:min(s + CH, R)]
            xc.add_(self.mlp(self.layernorm_mlp(xc)))
    _calls[0] += 1
    if os.environ.get("LEAN_LOG") == "1":
        now = time.time()
        if _calls[1] is not None and _calls[0] % 6 == 0:
            print(f"    [lean] ICL layer call {_calls[0]}  last {now - _calls[1]:.1f}s  rows={R:,}  "
                  f"mem={torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0:.2f} GB", flush=True)
        _calls[1] = now
    return x, kv_entry


def _lean_decoder_keys(self, train_emb):
    """Row-chunked `project_decoder_keys` (a per-row MLP and projection, so chunking is exact)."""
    B, N = train_emb.shape[:2]
    if N <= CH:
        return _orig_decoder_keys(self, train_emb)
    out = None
    with torch.no_grad():
        for s in range(0, N, CH):
            e = min(s + CH, N)
            k = _orig_decoder_keys(self, train_emb[:, s:e])
            if out is None:
                out = torch.empty((B, N) + tuple(k.shape[2:]), dtype=k.dtype, device=k.device)
            out[:, s:e] = k
            del k
    return out


def apply():
    A.ICLTransformerBlock.forward = _lean_forward
    A.MultiTaskHeads.project_decoder_keys = _lean_decoder_keys
