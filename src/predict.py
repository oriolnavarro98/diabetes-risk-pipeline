"""
predict.py - runs the model with a given input to produce
a diabetes risk prediction.
"""

import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import joblib

from .schemas import PredictRequest, PredictResponse

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

@lru_cache(maxsize=1)
def load_artifact():
    """
    Loads models + metadata once per process. (cached after first call)
    """

    model = joblib.load(MODEL_DIR / 'model.joblib')
    with open(MODEL_DIR / 'metadata.json') as f:
        metadata = json.load(f)
    
    return model, metadata

def run_predict(request: PredictRequest) -> PredictResponse:
    """
    Executes the predection based on a predict request with a set of features
    """
    model, metadata = load_artifact()
    feature_order = metadata['feature_names']

    # extract features from request in the same order as features are fed onto the model.
    x = np.array([[getattr(request, name) for name in feature_order]])

    # execute prediciton + extract probability
    pred = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][pred])

    # extract risk label based on prediction
    pred_label = metadata['risk_label_map'][str(pred)]

    return PredictResponse(
        risk_level=pred_label,
        probability=round(prob, 4),
        model_version=metadata['version'],
    )
