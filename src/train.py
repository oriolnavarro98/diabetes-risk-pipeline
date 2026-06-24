"""
train.py - produces versioned model artifact + metadata
Run from project root: python -m src.train
"""

from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
import time
import joblib
import json

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)
RISK_LABEL_MAP = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
TRAINDATA_PATH = Path(__file__).resolve().parent.parent / 'data/diabetes_data_raw.csv'
FEATURE_NAMES = [
    'blood_glucose',
    'physical_activity',
    'diet',
    'medication_adherence',
    'stress_level',
    'sleep_hours',
    'hydration_level',
    'bmi'
]
INITIAL_LABEL = 'risk_score'
TARGET_LABEL = 'risk'
SEED = 42

def risk_mapping(x:float) -> int:
    '''
    Function for mapping risk score to a class [0, 1, 2]
    '''
    if x < 30:
        return 0
    elif (x >= 30) and (x <= 60):
        return 1
    else:
        return 2

def load_training_data():
    """
    Import training data from data directory as a pandas dataFrame
    
    Output:
        X :: pd.DataFrame - predictive features points
        y :: pd.DataFrame - Source labels
    """
    # load dataset
    df = pd.read_csv(TRAINDATA_PATH)

    # apply mapping to new label to predict
    df[TARGET_LABEL] = df[INITIAL_LABEL].apply(risk_mapping) 

    # Extract features (x) + labels (y)
    X = df[FEATURE_NAMES].values
    y = df[TARGET_LABEL]

    return X, y

def train():
    """
    Model training function. It loads X and y using load_training_data and trains
    an XGBoost Model classifier to predict the diabetes risk based on a set of
    predetermined features.

    This method extracts a trained model (model.joblib) and its metadata (metadata.json)
    
    Output:
        None
    """
    X, y = load_training_data()

    """
    For performance metrics an honest estimate using 5 cross fold stratification 
    Training then is performed on all the data, since the size of the data set is
    limitted.
    """

    # model definition
    # define XGBoost classifier object
    clf = XGBClassifier(
        n_estimators = 150,
        max_depth = 4,
        learning_rate = 0.1,
        eval_metric = 'mlogloss', # logloss for multiclass
        random_state = SEED
    )

    # Stratified K-Fold definition
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # execute cross_validation
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)

    # extract metadat
    metrics = {
        "accuracy": round(accuracy_score(y, y_pred_cv), 4),
        "f1": round(f1_score(y, y_pred_cv, average='macro'), 4),
    }

    print(f'CV Metrics: {metrics}')

    # Execution of final training using the whole dataset
    clf.fit(X, y)

    version = time.strftime("%Y%m%d-%H%M%S")
    joblib.dump(clf, MODEL_DIR / "model.joblib")

    metadata = {
        'version': version,
        'feature_names': FEATURE_NAMES,
        'metrics': metrics,
        'risk_label_map': RISK_LABEL_MAP
    }

    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved -> {MODEL_DIR / 'model.joblib'}")
    print(f"Saved -> {MODEL_DIR / 'metadata.json'}")

if __name__ == '__main__':
    train()


