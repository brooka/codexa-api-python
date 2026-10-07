"""Query embeddings with EmbeddingGemma-300m.

A query must be embedded exactly as the packs were: the same int8 ONNX model, prompts, 256-dimension
truncation and text clean-up, so its vector lands in the same space as the stored ones.
"""

import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DIM = 256  # Matryoshka truncation of the model's 768 dimensions, as the packs store them
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

# Pointed Hebrew embeds poorly, so punctuation marks become spaces and vowel points and cantillation
# are dropped before embedding.
_HEBREW_PUNCTUATION = re.compile("[־׀׃׆]")
_HEBREW_POINTS = re.compile("[֑-ׇֽֿׁׂׅׄ]")


def clean_text(text: str) -> str:
    text = _HEBREW_POINTS.sub("", _HEBREW_PUNCTUATION.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip()


class Embedder:
    def __init__(self, model_dir: Path):
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"]
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(QUERY_PREFIX + clean_text(text))

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(DOCUMENT_PREFIX + clean_text(text))

    def _embed(self, text: str) -> np.ndarray:
        input_ids = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
        (pooled,) = self._session.run(
            ["sentence_embedding"], {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}
        )
        vector = pooled[0, :DIM]
        return vector / np.linalg.norm(vector)
