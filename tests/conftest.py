"""Shared fixtures: two tiny packs built on the fly, a fake embedder and a fake LLM, so the tests need
no model files, no network and no API keys."""

import hashlib
import json
import os
import sqlite3

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"  # set before the Langfuse client is first created

import numpy as np
import pytest
from fastapi.testclient import TestClient

from codexa_api.assistant import Assistant, Completion
from codexa_api.main import create_app
from codexa_api.packs import load_packs, to_int8
from codexa_api.search import Searcher

DIM = 64

VERSES = [
    ("Genesis 1:1", "In the beginning God created the heaven and the earth."),
    ("John 11:35", "Jesus wept."),
    ("2 Kings 2:23", "Little children mocked him, and said unto him, Go up, thou bald head."),
    ("Psalm 23:1", "The LORD is my shepherd; I shall not want."),
    ("Matthew 5:9", "Blessed are the peacemakers: for they shall be called the children of God."),
]
COMMENTARY = [
    ("Genesis 1:1", "Creation is the work of God alone: the heaven and the earth had a beginning."),
    ("Psalm 23:1", "The shepherd provides for every need of his flock."),
]


class FakeEmbedder:
    """Words hashed into a small vector, so texts that share words point the same way."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_document(text)

    def embed_document(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM)
        for word in text.lower().split():
            vector[int(hashlib.md5(word.strip(".,;:!?").encode()).hexdigest(), 16) % DIM] += 1
        return vector / (np.linalg.norm(vector) or 1)


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer: str = "God created the heaven and the earth [1]."):
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> Completion:
        self.prompts.append(prompt)
        return Completion(self.answer, input_tokens=120, output_tokens=12)


def build_pack(path, work_id: str, rows: list[tuple[str, str]]) -> None:
    """A pack with the real schema: texts, an FTS5 index and Int8 vectors in one blob."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE verses(ord INTEGER PRIMARY KEY, ref TEXT, book TEXT, chapter INTEGER, verse INTEGER,
                            text TEXT, lang TEXT);
        CREATE VIRTUAL TABLE verses_fts USING fts5(text, ref, tokenize='unicode61');
        CREATE TABLE vectors(id INTEGER PRIMARY KEY, data BLOB);
        CREATE TABLE meta(work_id TEXT, count INTEGER, dim INTEGER, model TEXT);
        """
    )
    embedder = FakeEmbedder()
    for ord, (ref, text) in enumerate(rows):
        db.execute("INSERT INTO verses(ord, ref, text, lang) VALUES (?, ?, ?, 'en')", (ord, ref, text))
        db.execute("INSERT INTO verses_fts(rowid, text, ref) VALUES (?, ?, ?)", (ord, text, ref))
    vectors = np.stack([to_int8(embedder.embed_document(text)) for _, text in rows])
    db.execute("INSERT INTO vectors VALUES (0, ?)", (vectors.tobytes(),))
    db.execute("INSERT INTO meta VALUES (?, ?, ?, 'fake')", (work_id, len(rows), DIM))
    db.commit()
    db.close()


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "mobile").mkdir()
    build_pack(tmp_path / "mobile" / "kjv.db", "kjv", VERSES)
    build_pack(tmp_path / "mobile" / "mhenry.db", "mhenry", COMMENTARY)
    manifest = {
        "packs": [
            {"work_id": "kjv", "file": "mobile/kjv.db", "type": "verse"},
            {"work_id": "mhenry", "file": "mobile/mhenry.db", "type": "commentary"},
        ]
    }
    (tmp_path / "packs.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def searcher(data_dir):
    return Searcher(load_packs(data_dir), FakeEmbedder())


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def assistant(searcher, llm):
    return Assistant(searcher, llm, k=3)


@pytest.fixture
def client(searcher, assistant):
    with TestClient(create_app(searcher, assistant)) as test_client:
        yield test_client
