from typing import Literal

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(
        ..., description="3 = normal, 6 = fixed defect, 7 = reversible defect"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    diagnosis: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
