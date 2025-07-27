import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset

from dataset.fm_model import FoundationModel
from inference import extract_features_batched


class EmbeddingCache(TensorDataset):
    """
    A dataset that stores embeddings and labels for images.
    Inherits from TensorDataset to allow easy access to embeddings and labels.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_paths: list[str] | None = None,
    ):
        """
        Initializes the EmbeddingCache with embeddings and labels.

        :param embeddings: Tensor of shape (N, D) where N is the number of images and D is the embedding dimension
        :param labels: Optional tensor of shape (N,) containing labels for the images
        :param image_paths: Optional list of image paths corresponding to the embeddings
        """
        super().__init__(embeddings, labels)
        self.embeddings = embeddings
        self.labels = labels if labels is not None else torch.tensor([])
        self.image_paths = image_paths

    @staticmethod
    def init_from_image_dataset(
        image_dataset: ImageFolder,
        model: FoundationModel,
        batch_size: int,
    ) -> "EmbeddingCache":
        """
        Initializes the EmbeddingCache with embeddings and labels.

        :param image_dataset: ImageFolder dataset containing images
        :param model: FoundationModel instance containing the model and processor
        :param batch_size: Batch size for processing images
        """
        full_tensor = torch.stack([img for img, _ in image_dataset], dim=0)
        assert full_tensor.dim() == 4  # batch tensor of shape (N, 3, H, W)
        embeddings = extract_features_batched(full_tensor, model, batch_size)
        image_paths = [img_path for img_path, _ in image_dataset.imgs]
        if hasattr(image_dataset, "targets"):
            return EmbeddingCache(
                embeddings, torch.tensor(image_dataset.targets), image_paths
            )
        else:
            return EmbeddingCache(embeddings, image_paths=image_paths)

    def __getitem__(self, index):
        """
        Returns the embedding and label for the given index.

        :param index: Index of the item to retrieve
        :return: Tuple of (image_path, embedding, label)
        """
        embedding, label = super().__getitem__(index)
        image_path = self.image_paths[index]
        return image_path, embedding, label

    def save(self, cache_path: str) -> None:
        """
        Saves the embedding cache to a file.

        :param embedding_cache: Dictionary mapping image paths to their corresponding embeddings
        :param cache_path: Path to save the cache file
        """
        torch.save(
            {
                "embeddings": self.embeddings,
                "labels": self.labels,
                "image_paths": self.image_paths,
            },
            cache_path,
        )

    @staticmethod
    def load_from_file(cache_path: str) -> "EmbeddingCache":
        """
        Loads the embedding cache from a file.

        :param cache_path: Path to the cache file
        :return: EmbeddingCache instance
        """
        dataset_dict = torch.load(cache_path)
        return EmbeddingCache(
            embeddings=dataset_dict["embeddings"],
            labels=dataset_dict.get("labels", None),
            image_paths=dataset_dict.get("image_paths", None),
        )
