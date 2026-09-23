import torch

def norm_chunk_into(residual_out, modulated_out, residual, gate, branch, weight, shift, scale, indices, eps):
    """Original row kernel writes final slices inside the eager attention island."""
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import _indexed_gate_rms_norm_scale_shift_kernel

    if not (residual.ndim == 2 and residual.shape[1] == 5376 and residual.shape[0] == 2996):
        raise ValueError("Norm slice must be [2996,5376]")
    tensors = (residual_out, modulated_out, residual, gate, branch, weight, shift, scale)
    if torch.is_grad_enabled() or not all(t.is_cuda and t.device == residual.device for t in tensors):
        raise ValueError("Norm slice is inference-only on a single CUDA device")
    if not all(tuple(t.shape) == tuple(residual.shape) and t.is_contiguous() and t.dtype == residual.dtype
               for t in (residual_out, modulated_out, branch)):
        raise ValueError("Norm slice branch/output contract mismatch")
    if residual.dtype != torch.bfloat16 or not residual.is_contiguous():
        raise ValueError("Norm slice requires contiguous BF16 residual")
    if not (weight.shape == (5376,) and weight.is_contiguous()
            and all(t.ndim == 2 and t.shape == gate.shape and t.shape[1] == 5376 and t.stride(1) == 1
                    for t in (gate, shift, scale))
            and indices.shape == (2996,) and indices.device == residual.device
            and indices.dtype in (torch.int32, torch.int64)):
        raise ValueError("Norm slice parameters/indices contract mismatch")
    # Caller passes the original validated combined_indices slice, preserving
    # the mapping to modality/timestep parameters and avoiding device readback.
    inputs = (residual, gate, branch, weight, shift, scale, indices)
    outputs = (residual_out, modulated_out)
    if (outputs[0].untyped_storage().data_ptr() == outputs[1].untyped_storage().data_ptr()
            or any(out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
                   for out in outputs for inp in inputs)):
        raise ValueError("Norm slice outputs must not alias each other or any input")
    _indexed_gate_rms_norm_scale_shift_kernel[(residual.shape[0],)](
        residual_out, modulated_out, residual, gate, branch, weight, shift, scale, indices,
        5376, eps, residual.stride(0), gate.stride(0), branch.stride(0),
        shift.stride(0), scale.stride(0), indices.stride(0), block_n=8192, num_warps=8,
    )
