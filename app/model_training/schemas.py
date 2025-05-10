from pydantic import BaseModel
from typing import Optional

class PatientInput(BaseModel):
    department: str
    municipality: str
    sex: str
    age: float
    marital_status: str
    education_level: int
    socioeconomic_status: int
    occupation: str
    is_smoker: int
    bpm_meds: int
    total_cholesterol: float
    bmi: float
    heart_rate: int
    glucose: float
    diabetes: int
    rural_area: int
    has_electricity: int
    has_water_supply: int
    has_sewage: int
    has_gas: int
    has_internet: int
    ethnicity: str
    family_history: int