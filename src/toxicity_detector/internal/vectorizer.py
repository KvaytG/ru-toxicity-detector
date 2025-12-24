import torch
import numpy as np
import pathlib
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from .preprocess import preprocess_text

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR.parent / "resources" / "rubert_tiny_onnx"
MODEL_NAME = "cointegrated/rubert-tiny2"

def _get_model_and_tokenizer():
    onnx_file = MODEL_PATH / "model.onnx"
    if not onnx_file.exists():
        print(f"ONNX model not found in {MODEL_PATH}. I'm starting the download and conversion...")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
        print("The model has been successfully saved locally.")
    else:
        model = ORTModelForFeatureExtraction.from_pretrained(str(MODEL_PATH), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    return model, tokenizer

_model, _tokenizer = _get_model_and_tokenizer()

def vectorize(text: str) -> np.ndarray:
    text = preprocess_text(text)
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=312
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    return embedding.numpy()
