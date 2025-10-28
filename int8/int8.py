import torch

def quantize_int8_asymmetric(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = int(round(qmin - min_val / scale))

    q_tensor = torch.clamp((tensor / scale + zero_point).round(), qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_int8_asymmetric(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)
