from int8 import quantize_int8_asymmetric, dequantize_int8_asymmetric


def quantize_model_int8(model):
    metadata = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            q_tensor, scale, zero_point = quantize_int8_asymmetric(param.data)
            param.data = q_tensor.float()  # Store as float for compatibility
            metadata[name] = {
                "scale": scale,
                "zero_point": zero_point,
                "original_dtype": str(param.dtype)
            }
    model._quant_metadata = metadata
    return model

def dequantize_model_int8(model):
    metadata = getattr(model, "_quant_metadata", None)
    if metadata is None:
        raise ValueError("No quantization metadata found in model.")

    for name, param in model.named_parameters():
        if param.requires_grad and name in metadata:
            meta = metadata[name]
            scale = meta["scale"]
            zero_point = meta["zero_point"]
            param.data = dequantize_int8_asymmetric(param.data, scale, zero_point).to(torch.float32)
    return model
