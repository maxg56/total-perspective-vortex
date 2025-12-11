"""
Model persistence utilities for EEG classification.

Provides functions to save and load trained models.
"""

import os
import logging
from typing import Tuple, Dict, Optional, Any
import joblib
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger(__name__)


def save_model(pipeline: Pipeline, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    path : str
        Path to save the model
    metadata : dict
        Additional metadata to save with the model
    """
    model_data = {
        'pipeline': pipeline,
        'metadata': metadata or {}
    }

    # Create directory if needed (handle both absolute and relative paths)
    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)

    # Use joblib for safer and more efficient serialization
    joblib.dump(model_data, path)

    logger.info(f"Model saved to: {path}")
    print(f"Model saved to: {path}")


def load_model(path: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load trained model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model

    Returns
    -------
    pipeline : Pipeline
        Trained sklearn pipeline
    metadata : dict
        Metadata saved with the model
    """
    # Use joblib for loading
    model_data = joblib.load(path)

    return model_data['pipeline'], model_data['metadata']
