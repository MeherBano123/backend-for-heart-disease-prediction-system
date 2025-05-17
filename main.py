
# main.py
from fastapi import FastAPI,HTTPException, Depends,Request,status, APIRouter, Security,Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse,FileResponse
from sqlmodel import SQLModel, Field, Session, create_engine , select
from typing import Optional
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased
from sqlalchemy import desc
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, EmailStr
import asyncio
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from typing import Optional, List
from datetime import datetime , date,timedelta
from sqlalchemy.sql import func
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
import traceback
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import ForeignKey
from jose import JWTError, jwt
from models import *
from schemas import *
from .env import SECRET_KEY,ALGORITHM,ACCESS_TOKEN_EXPIRE_MINUTES
import bcrypt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi_mail import FastMail, MessageSchema,ConnectionConfig
from starlette.requests import Request
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlmodel import select
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sqlmodel import Session as SQLModelSession
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import pickle
import os
from passlib.context import CryptContext
from reportlab.lib.units import inch
from predictions import preprocess_input
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
# from sqlalchemy.orm import Session
from sqlmodel import select
from sqlmodel import Session, select
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from sqlalchemy.future import select


# Rest of your FastAPI code...

app = FastAPI()
model = None
# Database engine setup
engine = create_engine("sqlite:///heartdisease.sqlite") 


# Create tables when application starts
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.on_event("startup")
def load_model():
    global model
    model_path = "xgboost_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError("Model file not found.")
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)
        if not hasattr(model_obj, "predict"):
            raise TypeError("Loaded object is not a trained model.")
        model = model_obj
    print("✅ ML model loaded successfully.")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# password reset 
RESET_PASSWORD_SECRET = URLSafeTimedSerializer(SECRET_KEY)
TOKEN_EXPIRATION_SECONDS = 3600  # 1 hour


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: EmailStr
    username: str
    first_name: str
    last_name: str
    role: str

class TokenData(BaseModel):
    username: Optional[str] = None

def get_session():
    with Session(engine) as session:
        yield session

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print("Validation error:", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
# Functions for password handling
def hash_password(password: str) -> str:
    # Generate a salt and hash the password
    if not isinstance(password, str):
        raise ValueError("Password must be a string")
        
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

    
# Function to authenticate user
def authenticate_user(email: str, password: str, session: Session):
    user = session.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user
def authenticate_user_with_security_answer(email: str, security_answer: str, session: Session):
    user = session.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not verify_password(security_answer, user.security_answer_hash):
        return False
    return user 
#function to create access_token

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
  
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        # Handle or log the exception appropriately
        raise ValueError(f"Failed to create access token: {str(e)}")


# decode token for logged in user 
def decode_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
        return token_data
    except jwt.PyJWTError:
        raise credentials_exception


security = HTTPBearer()
def get_current_user_id(credentials: HTTPAuthorizationCredentials = Security(security)) -> int:
    """Get user ID from Authorization header for use in Swagger UI"""
    try:
        # Extract user ID from the token
        # For simplicity, we're assuming the token is the user ID
        user_id = credentials.credentials
        return int(user_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    

def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        
        user = session.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


class UserInDB(User):
    hashed_password: str

def get_user(db, username: str):
    if username in db:
        user_data = db[username]
        return UserInDB(**user_data)
    

def get_db():
    with Session(engine) as session:
        yield session    
 # function to get current user 

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception

        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception

    user = get_user(get_db(), username=token_data.username)
    if user is None:
        raise credential_exception

    return user

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction APP"}

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},)


## CRUD apis for users model
@app.post("/users/registration")
def create_user( user_data: UserCreate, session: Session = Depends(get_session)):
    # Check if username already exists
    if session.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email already exists
    if session.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        # Hash the password properly
        password_hash = hash_password(user_data.password)
        #hash the security answer
        security_answer_case = user_data.security_answer.lower()
        # hash_security_answer = hash_password(security_answer_case)

        
        # Create a new User instance
        new_user = User(
            
            username=user_data.username,
            email=user_data.email,
            password_hash= password_hash, 
            first_name = user_data.first_name.title(),
            last_name = user_data.last_name.title(),
            role=user_data.role,
            security_question='What is your Birth City ?',
            security_answer_hash=security_answer_case,
            created_at=datetime.utcnow(),  # Set server-side
            last_login=datetime.utcnow() 
        )
        
        # Ensure timestamps are datetime objects
        if isinstance(new_user.created_at, str):
            new_user.created_at = datetime.fromisoformat(new_user.created_at.replace("Z", "+00:00"))
        if isinstance(new_user.last_login, str):
            new_user.last_login = datetime.fromisoformat(new_user.last_login.replace("Z", "+00:00"))
        
    
        # Print debug information
        print(f"Password hash: {password_hash}")
        print(f"User data: {new_user.__dict__}")
        
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        access_token = create_access_token(
        data={"sub": new_user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    
    except Exception as e:
        session.rollback()
        print(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    return {
    "access_token": access_token,
    "token_type": "bearer",
    "user_id": str(new_user.id),
    "email": new_user.email,
    "first_name": new_user.first_name,
    "last_name": new_user.last_name,
    "role": new_user.role,
    "username": new_user.username
}
   
    
@app.get("/users/get", response_model=List[User])
def read_users(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return users

@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# modify users 
@app.patch("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: User, session: Session = Depends(get_session)):
    db_user = session.get(User, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.dict(exclude_unset=True)
    for key, value in user_data.items():
        setattr(db_user, key, value)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

router = APIRouter()


# delete users 
@app.delete("/users/{user_id}")
def delete_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"ok": True}

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    secret_key: str = SECRET_KEY,
    algorithm: str = ALGORITHM
) -> str:
    """
    Creates a JWT access token with optional expiration time
    
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default expiration
    
    to_encode.update({"exp": expire})
    
    # Generate and return the token
    encoded_jwt = jwt.encode(
        to_encode,
        secret_key,
        algorithm=algorithm
    )
    return encoded_jwt


# user login api 
@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin, session: Session = Depends(get_session)):
    user = authenticate_user(form_data.email, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    session.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user.id),
        "email": user.email,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": user.role
        
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.post("/api/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    print("Received request:", request.dict())

    # Retrieve the user by email and plain-text security answer
    user = db.query(User).filter(
        User.email == request.email,
        User.security_answer_hash == request.security_answer
    ).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or security answer"
        )

    # Hash and update the new password
    user.hashed_password = pwd_context.hash(request.new_password)
    db.commit()

    return {"message": "Password reset successful"}


# api to create patient
@app.post("/patients/create", response_model=PatientResponse)
def create_patient(
    patient_data: PatientCreate,
    session: Session = Depends(get_session),
    # current_user: User = Depends(get_current_user)
):
    try:
        new_patient = Patient(
            **patient_data.dict(),
            created_at=datetime.utcnow(),
            recorded_at=datetime.utcnow()
        )
        session.add(new_patient)
        session.commit()
        session.refresh(new_patient)
        return new_patient
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating patient: {str(e)}")


@app.get("/patient/retrieve", response_model=List[PatientResponse])
def get_patients(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Get all patients for the current user"""
    patients = db.exec(
        select(Patient)
        .where(Patient.username == current_user_id)
        .offset(skip)
        .limit(limit)
    ).all()
    
    return patients

# retrieve all pateints associated with a specific user
@app.get("/patient/retrieve/{user_id}", response_model=List[PatientResponse])
def get_patient(
    user_id: int, 
    db: Session = Depends(get_db),
    # current_user_id: int = Depends(get_current_user_id)
):
    """Get all  patients by ID"""
    patients = db.execute(
        select(Patient).where(Patient.user_id == user_id)
    ).scalars().all()
    
    if not patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient of user: {user_id} not found"
        )
        
    return patients

@app.put("/patient/update/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int, 
    patient_update: PatientUpdate, 
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Update an existing patient record"""
    db_patient = db.get(Patient, patient_id)
    
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found"
        )
    
    # Ensure user can only update their own patient records
    if db_patient.username != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this patient record"
        )
    
    # Update patient data excluding None values
    update_data = patient_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_patient, key, value)
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.delete("/patient/delete/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Delete a patient record"""
    db_patient = db.get(Patient, patient_id)
    
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found"
        )
    
    # Ensure user can only delete their own patient records
    if db_patient.username != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this patient record"
        )
    
    db.delete(db_patient)
    db.commit()
    return None




@app.post("/predict/{patient_id}", response_model=Prediction)
def create_prediction(patient_id: int, session: Session = Depends(get_session)):
    try:
        # 1. Get patient - SQLModel compatible query
        statement = select(Patient).where(Patient.patient_id == patient_id)
        patient = session.exec(statement).scalar_one_or_none()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # 2. Extract features
        features = [
            patient.age,
            patient.blood_pressure,
            patient.cholesterol,
            patient.max_heart_rate,
            patient.oldpeak,
            str(patient.gender),  # Ensure string type
            str(patient.chest_pain_type),
            bool(patient.fasting_blood_sugar),
            str(patient.resting_ecg),
            bool(patient.exercise_angina),
            str(patient.st_slope)
        ]

        # 3. Preprocess features
        processed_features = preprocess_input(features)
        
        # 4. Predict
        prediction = model.predict([processed_features])[0]
        probability = model.predict_proba([processed_features])[0][1]

        # 5. Determine labels
        prediction_label = "Heart Disease" if prediction == 1 else "No Heart Disease"
        risk_level = "High" if prediction == 1 and probability >= 0.5 else "Low"

        # 6. Create Prediction record
        new_prediction = Prediction(
            patient_id=patient_id,
            risk_level=risk_level,
            prediction_label=prediction_label,
            model_used="XGBoost Classifier",
            confidence_score=float(np.float64(probability * 100).round(2))
        )
        
        session.add(new_prediction)
        session.commit()
        session.refresh(new_prediction)
        
        return new_prediction

    except Exception as e:
        session.rollback()
        
        print("Prediction error:", str(e))
        traceback.print_exc()  # Logs full stack trace
        raise HTTPException(status_code=500, detail=str(e))
    
# api to get latest prediction of a specific user 
@app.get("/predictions/patient/{patient_id}", response_model=Prediction)
def get_predictions_by_patient(patient_id: int, session: Session = Depends(get_session)):
    prediction = session.exec(
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(desc(Prediction.created_at))
        .limit(1)
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found for this patient")
    
    return prediction

## api to return prediction report




@app.get("/prediction/report/{prediction_id}")
def get_prediction_report(prediction_id: int, db: Session = Depends(get_db)):
# Fetch prediction
    prediction = db.execute(
    select(Prediction).where(Prediction.prediction_id == prediction_id)
    ).scalars().first()


    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Fetch associated patient
    patient = db.execute(
        select(Patient).where(Patient.patient_id == prediction.patient_id)
    ).scalars().first()

    if not patient:
        raise HTTPException(status_code=404, detail="Associated patient not found")

    # File setup
    filename = f"prediction_report_{prediction_id}.pdf"
    filepath = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)

    # Create PDF
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    # Backgrounds
    c.setFillColor(HexColor("#f7fafc"))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    c.setFillColor(HexColor("#e2e8f0"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(HexColor("#A10000"))
    c.drawCentredString(width / 2, height - 50, "Heart Disease Prediction System")

    # Set text color
    c.setFont("Helvetica", 12)
    c.setFillColor(HexColor("#374151"))
    y = height - 100
    line_gap = 20

    # Patient Info Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Patient Information:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Name: {patient.full_name}")
    y -= line_gap
    c.drawString(60, y, f"Age: {patient.age}")
    y -= line_gap
    c.drawString(60, y, f"Gender: {patient.gender}")
    y -= line_gap
    c.drawString(60, y, f"Email: {patient.email}")

    # Prediction Info
    y -= 2 * line_gap
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Prediction Details:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Prediction ID: {prediction.prediction_id}")
    y -= line_gap
    c.drawString(60, y, f"Prediction Label: {prediction.prediction_label}")
    y -= line_gap
    c.drawString(60, y, f"Risk Level: {prediction.risk_level}")
    y -= line_gap
    c.drawString(60, y, f"Model Used: {prediction.model_used}")
    y -= line_gap

    confidence_str = prediction.confidence_score
    try:
        confidence_float = float(str(confidence_str).strip('%'))
        c.drawString(60, y, f"Confidence Score: {confidence_float:.2f}%")
    except ValueError:
        c.drawString(60, y, f"Confidence Score: {confidence_str}")
    y -= line_gap

    created_at = prediction.created_at.strftime('%Y-%m-%d %H:%M:%S') if prediction.created_at else 'N/A'
    c.drawString(60, y, f"Created At: {created_at}")

    # Recommendations
    y -= 2 * line_gap
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Recommendations for a Healthy Heart:")
    c.setFont("Helvetica", 12)
    for rec in [
        "1. Don't smoke or use tobacco.",
        "2. Get moving: Aim for at least 30 to 60 minutes of activity daily.",
        "3. Eat a heart-healthy diet including:",
        "   - Vegetables and fruits",
        "   - Beans or other legumes",
        "   - Lean meats and fish",
        "   - Low-fat or fat-free dairy foods",
        "   - Whole grains",
        "   - Healthy fats such as olive oil and avocado"
    ]:
        y -= line_gap
        c.drawString(60, y, rec)

    # Disclaimer
    y -= 2 * line_gap
    c.setFont("Helvetica-Oblique", 11)
    c.setFillColor(HexColor("#374151"))
    c.drawString(50, y, "Note: Consult a cardiologist for treatment and diagnosis. ML Systems can make mistakes.")

    c.save()

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/pdf'
    )
    
