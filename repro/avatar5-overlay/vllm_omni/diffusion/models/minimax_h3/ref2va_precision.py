"""Request-selected attention precision; dispatch only inside opaque CUDA ops."""
import json
import functools
from collections import Counter

POLICY = 'fast'
COUNTS = Counter()
BF16_STEPS = {'fast': 0, 'sage_first_bf16': 1, 'sage_first_two_bf16': 2, 'all_bf16': 4}


def profile_capture(function):
    @functools.wraps(function)
    def wrapped(self, request):
        capture = (request.sampling_params.extra_args or {}).get('nsys_capture', False)
        if type(capture) is not bool:
            raise ValueError('nsys_capture must be a boolean')
        capture = capture and getattr(self, '_dit_rank', 0) == 0
        if capture:
            import torch
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        try:
            return function(self, request)
        finally:
            if capture:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
    return wrapped


def begin(policy):
    global POLICY
    if policy not in BF16_STEPS:
        raise ValueError('Unsupported Ref2VA attention precision policy')
    POLICY = policy
    COUNTS.clear()


def attention_mode():
    from vllm_omni.diffusion.forward_context import get_forward_context
    ctx = get_forward_context()
    if ctx.total_denoise_steps != 4 or ctx.denoise_step_idx not in range(4):
        raise RuntimeError('Precision audit requires explicit four-step execution')
    mode = 'bf16' if ctx.denoise_step_idx < BF16_STEPS[POLICY] else 'sage'
    COUNTS[(ctx.denoise_step_idx,mode)] += 1
    return mode


def finish():
    import torch
    expected = Counter({(i,'bf16' if i < BF16_STEPS[POLICY] else 'sage'):50 for i in range(4)})
    if COUNTS != expected:
        raise RuntimeError(f'Actual attention dispatch differs: {COUNTS}')
    print('REF2VA_ATTENTION_PRECISION '+json.dumps(dict(rank=torch.distributed.get_rank(),
        policy=POLICY,steps=[dict(step=i+1,mode=mode,calls=n) for (i,mode),n in sorted(COUNTS.items())],status='PASS')),flush=True)
