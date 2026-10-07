"""Checks against the real packs and model. Skipped when this checkout doesn't have them."""

import numpy as np
import pytest

from codexa_api.config import get_settings
from codexa_api.embedder import Embedder
from codexa_api.packs import load_packs, to_int8
from codexa_api.search import Searcher

settings = get_settings()
pytestmark = pytest.mark.skipif(
    not (settings.data_dir / "packs.json").exists()
    or not (settings.model_dir / "onnx" / "model_quantized.onnx").exists(),
    reason="needs the built packs and the EmbeddingGemma model",
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(settings.model_dir)


def test_query_embeddings_match_the_stored_vectors(embedder):
    pack = load_packs(settings.data_dir, ["kjv"])[0]
    for position in (0, 12345):
        mine = to_int8(embedder.embed_document(pack.hit(position, 0).text)).astype(float)
        stored = pack.vectors[position].astype(float)
        assert mine @ stored / (np.linalg.norm(mine) * np.linalg.norm(stored)) > 0.99


def test_hybrid_search_finds_a_described_verse(embedder):
    searcher = Searcher(load_packs(settings.data_dir, ["kjv", "bbe"]), embedder)
    hits = searcher.search("the verse where kids make fun of a bald man and get mauled by bears", limit=5)
    assert {"2 Kings 2:23", "2 Kings 2:24"} & {h.ref for h in hits}
