"""
schemas.py - API validation schema before feeding input into the model
"""

import math
from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    """
    Validation of predict request parameters
    """
    blood_glucose: float
    physical_activity: float
    diet: int
    medication_adherence: int
    stress_level: int
    sleep_hours: float
    hydration_level: int
    bmi: float

    @field_validator("*")
    @classmethod
    def no_nan_or_inf(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Feature values must be finite")
        return v
    
class PredictResponse(BaseModel):
    """
    Validation of predict response
    """
    risk_level: str # the value will be converted with reference on metadata
    probability: float
    model_version: str