"""The SQLite packs: the searchable texts.

One file per work (a translation, a commentary, a dictionary): its texts, an FTS5 keyword index, and
every text's embedding stored as Int8 in one blob. packs.json lists the packs with their type.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = {
    "kjv": "KJV",
    "web": "WEB",
    "bbe": "Bible in Basic English",
    "sblgnt": "Greek (SBLGNT)",
    "wlc": "Hebrew (WLC)",
    "mhenry": "Matthew Henry",
    "jfb": "Jamieson, Fausset & Brown",
    "gill": "John Gill",
    "clarke": "Adam Clarke",
    "kd": "Keil & Delitzsch",
    "easton": "Easton's Bible Dictionary",
    "catena1": "Church Fathers",
    "catena2": "Church Fathers",
    "catena3": "Church Fathers",
    "confessions": "Creeds & Confessions",
    "barnes": "Barnes' Notes (NT)",
    "tdavid": "Treasury of David (Spurgeon)",
    "scofield": "Scofield Reference Notes",
    "classics": "Classic Christian Works",
}

_WORD = re.compile(r"[^\W_]+")  # runs of letters and digits


def to_int8(vector: np.ndarray) -> np.ndarray:
    """A normalised vector (components in [-1, 1]) as Int8, the way the packs store theirs.

    Halves round up (floor(x + 0.5)), as in the stored vectors; -128 is avoided to keep the range symmetric.
    """
    return np.clip(np.floor(vector * 127 + 0.5), -127, 127).astype(np.int8)


@dataclass(frozen=True)
class Hit:
    work_id: str
    label: str
    type: str
    ref: str
    text: str
    score: float


class Pack:
    def __init__(self, path: Path, work_id: str, type: str = "verse"):
        self.work_id = work_id
        self.type = type
        self.label = LABELS.get(work_id, work_id)
        self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        count, dim = self._db.execute("SELECT count, dim FROM meta").fetchone()
        (blob,) = self._db.execute("SELECT data FROM vectors WHERE id = 0").fetchone()
        self.vectors = np.frombuffer(blob, dtype=np.int8).reshape(count, dim)

    def keyword(self, query: str, k: int) -> list[int]:
        """Positions of the best BM25 matches for any of the query's words, best first."""
        words = _WORD.findall(query.lower())
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self._db.execute(
            "SELECT rowid FROM verses_fts WHERE verses_fts MATCH ? ORDER BY rank LIMIT ?", (match, k)
        )
        return [row[0] for row in rows]

    def semantic(self, query: np.ndarray, k: int) -> list[tuple[int, int]]:
        """(position, score) of the k nearest vectors, best first. Score is the Int8 dot product."""
        scores = np.matmul(self.vectors, query, dtype=np.int32)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), int(scores[i])) for i in top]

    def hit(self, position: int, score: float) -> Hit:
        ref, text = self._db.execute("SELECT ref, text FROM verses WHERE ord = ?", (position,)).fetchone()
        return Hit(self.work_id, self.label, self.type, ref, text, score)


def load_packs(data_dir: Path, work_ids: list[str] | None = None) -> list[Pack]:
    """The packs listed in packs.json, in that order, or in the order of `work_ids` when given."""
    manifest = json.loads((data_dir / "packs.json").read_text())
    entries = {p["work_id"]: p for p in manifest["packs"]}
    wanted = work_ids or list(entries)
    missing = [w for w in wanted if w not in entries]
    if missing:
        raise ValueError(f"not in packs.json: {', '.join(missing)}")
    return [Pack(data_dir / entries[w]["file"], w, entries[w].get("type", "verse")) for w in wanted]
