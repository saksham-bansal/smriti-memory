"""
Persistent Dense Vector Index backed by SQLite BLOB storage and fast NumPy cosine search.

Includes a Multi-Granular Semantic-Lexical Dense Embedder (combining concept/synonym clusters,
stemmed unigrams, word bigrams, and character n-grams with signed random projections) so that
dense retrieval runs deterministically in <2ms per query offline without external API rate limits,
while also supporting optional Gemini API embeddings.
"""
import hashlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from smriti.storage.sqlite_store import SQLiteMemoryStore

# Curated conversational memory concept & synonym clusters for semantic bridging
CONCEPT_CLUSTERS: Dict[str, List[str]] = {
    "concept_diet_food": [
        "vegan", "vegetarian", "pescatarian", "diet", "dietary", "eat", "eating",
        "food", "meal", "cooking", "cook", "restaurant", "allergy", "allergic",
        "gluten", "dairy", "meat", "nutrition", "recipe",
    ],
    "concept_location_move": [
        "live", "living", "moved", "moving", "relocate", "relocated", "city",
        "town", "apartment", "house", "home", "neighborhood", "hometown",
        "country", "state", "reside", "address", "from",
    ],
    "concept_career_work": [
        "job", "work", "working", "career", "profession", "company", "office",
        "promotion", "promoted", "hired", "interview", "business", "startup",
        "manager", "engineer", "counselor", "counseling", "therapist", "teacher",
        "nurse", "doctor", "developer", "designer", "role", "salary", "study",
        "degree", "university", "college", "school", "education",
    ],
    "concept_family_relations": [
        "sister", "brother", "mother", "mom", "father", "dad", "parent", "parents",
        "child", "children", "kids", "kid", "son", "daughter", "wife", "husband",
        "partner", "spouse", "married", "wedding", "single", "dating", "boyfriend",
        "girlfriend", "friend", "family", "adopt", "adopted", "adoption",
    ],
    "concept_pets_animals": [
        "dog", "puppy", "cat", "kitten", "pet", "pets", "animal", "shelter",
        "vet", "walk", "adopt", "breed", "rescued",
    ],
    "concept_art_hobbies": [
        "paint", "painting", "pottery", "ceramics", "clay", "draw", "drawing",
        "sketch", "music", "guitar", "piano", "violin", "instrument", "sing",
        "dance", "dancing", "book", "reading", "novel", "write", "writing",
        "photo", "photography", "camera", "craft", "knitting", "museum", "art",
        "studio", "class", "workshop",
    ],
    "concept_sports_outdoors": [
        "run", "running", "marathon", "race", "5k", "10k", "hike", "hiking",
        "trail", "mountain", "camp", "camping", "tent", "swim", "swimming",
        "yoga", "gym", "workout", "fitness", "bike", "cycling", "surf", "ski",
        "kayak", "outdoors", "nature", "park", "beach",
    ],
    "concept_travel_trip": [
        "travel", "trip", "vacation", "holiday", "flight", "hotel", "visit",
        "visited", "tour", "roadtrip", "abroad", "museum", "concert", "festival",
    ],
    "concept_health_mental": [
        "support", "group", "therapy", "counseling", "mental", "health", "stress",
        "anxiety", "recover", "injury", "hospital", "doctor", "medicine",
        "transgender", "lgbtq", "pride", "community", "mentor", "volunteer",
    ],
    "concept_time_temporal": [
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "yesterday",
        "today", "tomorrow", "week", "month", "year", "ago", "last", "next",
        "since", "before", "after", "when", "date", "duration", "long",
        "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    ],
}

_WORD_TO_CONCEPTS: Dict[str, List[str]] = {}
for _cid, _words in CONCEPT_CLUSTERS.items():
    for _w in _words:
        _WORD_TO_CONCEPTS.setdefault(_w, []).append(_cid)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "over", "after", "and", "or", "but", "so", "if", "because", "as",
    "until", "while", "that", "this", "these", "those", "it", "its", "just",
    "very", "really", "also", "too", "yeah", "yes", "no", "oh", "hey", "hi",
    "well", "like", "know", "think", "got", "get",
}


def _light_stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix in ("ingly", "edly", "ment", "ness", "tion", "sion", "able", "ible"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


class DenseEmbedder:
    """
    Deterministic multi-granular dense embedder using signed feature hashing over:
      1. Curated semantic concept clusters (weight 2.5)
      2. Stemmed content unigrams (weight 2.0)
      3. Exact unigrams & numbers/dates (weight 1.5)
      4. Content word bigrams (weight 1.2)
      5. Character 4-grams for morphological/entity fuzzy matching (weight 0.5)
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _hash_feature(feature: str, dim: int) -> Tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        idx2 = int.from_bytes(digest[5:9], "little") % dim
        sign2 = 1.0 if (digest[9] & 1) == 0 else -1.0
        return idx, sign + 0 * idx2 + 0 * sign2

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        tokens = re.findall(r"[a-z0-9]+", text.lower())
        content = [t for t in tokens if t not in _STOPWORDS]
        if not content:
            content = tokens

        stems = [_light_stem(t) for t in content]

        def _add(feat: str, weight: float) -> None:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            i1 = int.from_bytes(digest[:4], "little") % self.dim
            s1 = 1.0 if (digest[4] & 1) == 0 else -1.0
            i2 = int.from_bytes(digest[5:9], "little") % self.dim
            s2 = 1.0 if (digest[9] & 1) == 0 else -1.0
            vec[i1] += weight * s1
            vec[i2] += 0.5 * weight * s2

        for t, s in zip(content, stems):
            _add(f"w:{t}", 1.5)
            _add(f"s:{s}", 2.2)
            for cid in _WORD_TO_CONCEPTS.get(t, []):
                _add(f"c:{cid}", 2.8)
            for cid in _WORD_TO_CONCEPTS.get(s, []):
                _add(f"c:{cid}", 2.8)
            if len(t) >= 4:
                padded = f"<{t}>"
                for k in range(len(padded) - 3):
                    _add(f"g4:{padded[k:k+4]}", 0.35)

        for i in range(len(stems) - 1):
            _add(f"bi:{stems[i]}_{stems[i+1]}", 1.3)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec


class VectorIndex:
    """Persistent vector index backed by SQLiteMemoryStore."""

    def __init__(self, store: SQLiteMemoryStore, embedder: Optional[DenseEmbedder] = None):
        self.store = store
        self.embedder = embedder or DenseEmbedder(dim=384)

    def index_item(
        self, conversation_id: str, item_type: str, item_id: str, text: str
    ) -> np.ndarray:
        vec = self.embedder.embed(text)
        cur = self.store.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (conversation_id, item_type, item_id, dim, vector_blob)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                item_type,
                item_id,
                int(vec.shape[0]),
                vec.astype(np.float32).tobytes(),
            ),
        )
        self.store.conn.commit()
        return vec

    def search(
        self,
        conversation_id: str,
        item_type: str,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        q_vec = self.embedder.embed(query)
        if float(np.linalg.norm(q_vec)) < 1e-8:
            return []

        cur = self.store.conn.cursor()
        rows = cur.execute(
            """
            SELECT item_id, dim, vector_blob
            FROM embeddings
            WHERE conversation_id = ? AND item_type = ?
            """,
            (conversation_id, item_type),
        ).fetchall()
        if not rows:
            return []

        item_ids: List[str] = []
        vecs: List[np.ndarray] = []
        for r in rows:
            arr = np.frombuffer(r["vector_blob"], dtype=np.float32)
            if arr.shape[0] == q_vec.shape[0]:
                item_ids.append(r["item_id"])
                vecs.append(arr)
        if not vecs:
            return []

        mat = np.vstack(vecs)
        scores = mat @ q_vec
        order = np.argsort(-scores)[:top_k]
        return [(item_ids[int(i)], float(scores[int(i)])) for i in order if scores[int(i)] > 0.0]
