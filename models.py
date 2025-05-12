from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP
from sqlalchemy.sql import func
from pydantic import BaseModel  
from sqlmodel import SQLModel, Field
from pydantic import EmailStr
from datetime import datetime,date,timezone
from typing import Optional

from passlib.context import CryptContext
from typing import Optional
from sqlmodel import SQLModel, Field, Session
from pydantic import validator



pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



# Define User model

class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True,index=True)
    username: str = Field(max_length=50, unique=True, nullable=False)
    password_hash: str = Field(max_length=256, nullable=False)
    first_name: str = Field(max_length=50)
    last_name: str = Field(max_length=50)
    email: EmailStr = Field(max_length=100, unique=True, nullable=False)
    role: str = Field(max_length=20, nullable=False)
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )
    last_login: Optional[datetime] = None
    is_active: bool = Field(default=True)

    @validator('created_at', 'last_login', pre=True)
    def parse_datetime(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("Invalid datetime format. Use ISO format.")
        return value
    
    def verify_password(self, plain_password: str):
        return pwd_context.verify(plain_password, self.hashed_password)
    


# Updated Patient Model 
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import date, datetime
from sqlalchemy.sql import func

class Patient(SQLModel, table=True):
    __tablename__ = "patients"

    patient_id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id", nullable=False)  #'id' is the primary key in users table

    full_name: str = Field(max_length=50, nullable=False)
    age: int = Field(nullable=False)
    date_of_birth: date = Field(nullable=False)
    contact_number: str = Field(max_length=15)
    gender: str = Field(max_length=10, nullable=False)
    email: EmailStr = Field(max_length=100)

    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )

    blood_pressure: int
    cholesterol: int
    fasting_blood_sugar: bool
    resting_ecg: str 
    max_heart_rate: int
    chest_pain_type: str 
    exercise_angina: bool
    oldpeak: float
    st_slope: str = Field(max_length=20)
    num_major_vessels: int
    thalassemia: str = Field(max_length=20)

    recorded_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )

    # Optional relationship (if using SQLAlchemy ORM features)
    # user: Optional["User"] = Relationship(back_populates="patients")



# Prediction Model (unchanged)
class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"
    prediction_id: Optional[int] = Field(default=None, primary_key=True,index=True  )
    patient_id: int = Field(foreign_key="patients.patient_id",nullable=True) 
    prediction_label:str = Field(max_length= 25,nullable = False)
    risk_level: str = Field(max_length=20, nullable=False)
    model_used: str = Field(max_length=50, nullable=False)
    confidence_score:float = Field(nullable=False)
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )


