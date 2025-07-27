import torch

from scipy.cluster.vq import kmeans2
from typing import Callable

from dataset.cached_embedding import EmbeddingCache

SEED = 42
torch.manual_seed(SEED)


def kmeans_clustering(
    embedding_cache: EmbeddingCache,
    n_clusters: int,
    max_iter: int = 100,
    min_change: float = 1e-5,
) -> tuple[
    Callable[[torch.FloatTensor], torch.IntTensor], torch.IntTensor, torch.FloatTensor
]:
    """
    Perform k-means clustering on the embeddings in the cache.

    :param embedding_cache: EmbeddingCache containing embeddings and labels
    :param n_clusters: Number of clusters to form
    :param model: FoundationModel instance containing the model and processor
    :param max_iter: Maximum number of iterations for k-means
    :param min_change: Minimum change in centroids to continue iterating
    :return: predict_fn, list of training predicted labels, list of cluster centroids
    """
    embeddings = embedding_cache.embeddings.cpu().numpy()

    # Perform k-means clustering
    centroids, labels = kmeans2(
        embeddings,
        n_clusters,
        iter=max_iter,
        minit="points",
        thresh=min_change,
        rng=SEED,
    )
    centroids = torch.tensor(centroids, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    def classify_embeddings(embeddings: torch.FloatTensor) -> torch.IntTensor:
        # move to embedding device to allow for GPU tensors
        # return device is the same as embeddings device
        distances = torch.cdist(embeddings, centroids.to(embeddings.device))
        return torch.argmin(distances, dim=1)

    return classify_embeddings, labels, centroids
