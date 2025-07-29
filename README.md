# Pathology foundation models

Interface for calling foundation models for pathology image analysis.

## Installing package

To install the package in development mode:

```bash
pip install -e .
```

Or to install with development dependencies:

```bash
pip install -e ".[dev]"
```

For Jupyter notebook support:

```bash
pip install -e ".[notebook]"
```

## Running experiments

Experiments should be placed under the `experiments` folder. After installing the package, you can import modules directly:

```python
from dataset.loader import load_foundation_model
from dataset.cached_embedding import EmbeddingCache
from unsupervised.kmeans import kmeans_clustering
```