import pathlib
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from .constants import MAX_LEN
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

def _mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def vectorize_batch(texts: list[str]) -> torch.Tensor:
    texts = [preprocess_text(t) if t else "" for t in texts]
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    embeddings = _mean_pooling(outputs, inputs["attention_mask"])
    return embeddings

def vectorize(text: str) -> torch.Tensor:
    return vectorize_batch([text])[0]
