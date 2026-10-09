"""
Key Feature Embedder: Generates independent embeddings for each key feature.
Strictly avoids merging or concatenating features into a single query.
"""

from app.embedder import Embedder
from app.keyfeature_searchflow.models import KeyFeatureItem


class KeyFeatureEmbedder:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()

    def embed_features(
        self, features: list[KeyFeatureItem]
    ) -> list[tuple[KeyFeatureItem, list[float]]]:
        """
        Embed EACH feature as an independent vector.
        Each feature maintains its own separate embedding.
        Features are NEVER concatenated or merged.
        """
        if not features:
            return []

        # Extract independent feature texts - each text is distinct
        feature_texts = [f.feature.strip() for f in features]

        # Use the embedder to get one vector per feature text
        vectors = self.embedder.embed_texts(feature_texts)

        if len(vectors) != len(features):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(features)}, got {len(vectors)}"
            )

        # Pair each feature item with its own independent embedding vector
        return list(zip(features, vectors))

