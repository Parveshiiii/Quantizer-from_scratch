import torch
from transformers import AutoModel, AutoTokenizer
from int8 import quantize_uint8_asymmetric, dequantize_uint8_asymmetric
from wrap import quantize_model_int8, dequantize_model_int8  # Your wrapper functions

model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


quantized_model = quantize_model_int8(model)


save_path = "bert_quantized.pt"
torch.save({
    "model_state_dict": quantized_model.state_dict(),
    "quant_metadata": quantized_model._quant_metadata
}, save_path)
print(f"Quantized model saved to {save_path}")


def load_and_dequantize_model(path, model_class=AutoModel, model_name="bert-base-uncased"):
    checkpoint = torch.load(path)
    model = model_class.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model._quant_metadata = checkpoint["quant_metadata"]
    model = dequantize_model_int8(model)
    return model


