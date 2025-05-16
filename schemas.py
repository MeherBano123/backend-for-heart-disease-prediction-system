# app/schemas.py
from pydantic import BaseModel,Field,EmailStr
from datetime import datetime,date
from typing import Optional,List
from pydantic import BaseModel, EmailStr,validator
from enum import Enum
from datetime import date

class ECGType(str, Enum):
    normal = "normal"
    stt_abnormality = "stt_abnormality"
    lv_hypertrophy = "lv_hypertrophy"

class ChestPainType(str, Enum):
    asymptomatic = "asymptomatic"
    atypical_angina = "atypical angina"
    non_anginal = "non-anginal"
    typical_angina = "angina"

class STSlope(str, Enum):
    flat = "flat"
    upsloping = "upsloping"
    downsloping = "downsloping"

class ThalassemiaType(str, Enum):
    normal = "normal"
    fixed_defect = "fixed defect"
    reversable_defect = "reversable defect"
    missing = "Missing"

##for handling lower case nad title case
@validator(
        'resting_ecg', 'chest_pain_type', 'st_slope', 'thalassemia',
        pre=True, allow_reuse=True
    )
def normalize_case(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name :str
    last_name:str
    role: Optional[str] = None
    security_answer: str  

class UserInDB(BaseModel):
    email: str
    username: str
    first_name: str
    last_name: str
    role: str
    password_hash: str
    security_answer_hash: str  # Hashed version stored in DB

class UserResponse(BaseModel):
    id: int
    username: str
    email:EmailStr
    security_question: str
    security_answer: str  
    model_config = {
        "from_attributes": True  
    }


## patient request and response model
class PatientCreate(BaseModel):
    full_name: str
    user_id:int
    age: int
    date_of_birth: date
    contact_number: str
    gender: str
    email: EmailStr
    blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: bool
    resting_ecg: ECGType
    max_heart_rate: int
    chest_pain_type: ChestPainType
    exercise_angina: bool
    oldpeak: float
    st_slope: STSlope
    num_major_vessels: int
    thalassemia: ThalassemiaType
    oldpeak:float
    

class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    date_of_birth: Optional[date] = None
    contact_number: Optional[str] = None
    gender: Optional[str] = None
    email: Optional[EmailStr] = None
    blood_pressure: Optional[float] = None
    cholesterol: Optional[int] = None
    fasting_blood_sugar: Optional[bool] = None
    resting_ecg: Optional[str] = None
    max_heart_rate: Optional[int] = None
    chest_pain_type: Optional[str] = None
    exercise_angina: bool
    oldpeak: Optional[float] = None
    st_slope: Optional[str] = None
    num_major_vessels: Optional[int] = None
    thalassemia: Optional[str] = None

class PatientResponse(BaseModel):
    patient_id: int
    user_id:int
    full_name: str
    age: int
    date_of_birth: date
    contact_number: str
    gender: str
    email: EmailStr
    created_at: datetime
    blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: bool
    resting_ecg: str
    max_heart_rate: int
    chest_pain_type: str
    exercise_angina: bool
    oldpeak: float
    st_slope: str
    num_major_vessels: int
    thalassemia: str
    recorded_at: datetime



#  prediction create model
class PredictionCreate(BaseModel):
    patient_id: int
    prediction_label: str = Field(..., max_length=25)
    risk_level: str = Field(..., max_length=20)
    model_used: str = Field(..., max_length=50)
    confidence_score:float


# Response model (includes ID and timestamp)
class PredictionResponse(BaseModel):
    prediction_id: int
    patient_id: int
    prediction_label:str
    risk_level:str
    model_used: str
    confidence_score:float 
    created_at:datetime

    

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class EmailSchema(BaseModel):
   email: List[EmailStr]

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    security_answer: str 
    new_password: str 

class VerifySecurityAnswerRequest(BaseModel):
    email: EmailStr
    security_answer: str
    new_password: str
