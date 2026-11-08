"""CUDA-graph replay of the incremental flow step (GatedCursorLM / LiuBaiH .step_velocity).

Eager step_velocity of the 3B / 5B LiuBaiH is host-bound: ~130 ms to enqueue 32 layers whether the batch has 1 or 64 rows
(2026-11-05, /tmp/genspeed/bench.py). One graph replay enqueues the whole step at once.

step_velocity only READS the cache (F rows never write S), but step_hidden reassigns cache[l]["S" / "p1" / "p2"] to new
tensors every block, so the graph runs on static copies: load() once per block (cache + window + advance + previous block),
then one __call__ per flow step. Shapes are fixed at capture: drop the object and build a new one when the row count changes.
"""
import torch


class GraphedVelocity:
    def __init__(s, M, cache, xt, t, lyr, a, zprev, dtype=torch.bfloat16):
        s.M, s.dtype, s.R = M, dtype, xt.shape[0]
        s.cache = [{k: v.clone() for k, v in cl.items()} for cl in cache]
        s.xt, s.t, s.lyr, s.a, s.zprev = xt.clone(), t.clone(), lyr.clone(), a.clone(), zprev.clone()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                                                # warm-up off the capture stream (cuBLAS handles)
            for _ in range(2): s._run()
        torch.cuda.current_stream().wait_stream(st)
        s.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g): s.out = s._run()

    def _run(s):
        with torch.autocast("cuda", dtype=s.dtype, cache_enabled=False):
            return s.M.step_velocity(s.xt, s.t, s.lyr, s.a, s.cache, s.zprev).float()

    def load(s, cache, lyr, a, zprev):
        """per block, after step_hidden: copy the live cache and this block's conditions into the static inputs"""
        for d, c in zip(s.cache, cache):
            for k, v in d.items(): v.copy_(c[k])
        s.lyr.copy_(lyr); s.a.copy_(a); s.zprev.copy_(zprev)

    def __call__(s, xt, t):
        """-> v [R,D] fp32; a view of the graph's output buffer, overwritten by the next call"""
        s.xt.copy_(xt); s.t.copy_(t); s.g.replay(); return s.out
