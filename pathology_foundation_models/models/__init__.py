# expose public functions

from models.config import FoundationModelEnum, get_embedding_dim
from models.fm_model import FoundationModel, load_foundation_model
from models.inference import extract_features, extract_features_from_dataset
