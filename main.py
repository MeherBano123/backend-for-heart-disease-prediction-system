
# main.py
from fastapi import FastAPI,HTTPException, Depends,Request,status, APIRouter, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse,FileResponse
from sqlmodel import SQLModel, Field, Session, create_engine , select
from typing import Optional
from sqlalchemy.orm import aliased
from sqlalchemy import desc
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, EmailStr
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from typing import Optional, List
from datetime import datetime , date,timedelta
from sqlalchemy.sql import func
import traceback
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import ForeignKey
from jose import JWTError, jwt
from models import *
from schemas import *
import bcrypt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
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
# Security configurations
SECRET_KEY = "926138007fe6ff3b7e6e296b43dc50c9ba15d9308b541e7d5af142bf9242ee23"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 45


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# password reset 
RESET_PASSWORD_SECRET = URLSafeTimedSerializer(SECRET_KEY)
TOKEN_EXPIRATION_SECONDS = 3600  # 1 hour


# SMTP server settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "cardioscandetectionsystem@gmail.com"
SMTP_PASSWORD = "uqkz hbca pbme xllm"  
FROM_EMAIL = "cardioscandetectionsystem@gmail.com"

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
def create_user(user_data: UserCreate, session: Session = Depends(get_session)):
    # Check if username already exists
    if session.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email already exists
    if session.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        # Hash the password properly
        password_hash = hash_password(user_data.password)
        
        # Create a new User instance
        new_user = User(
            
            username=user_data.username,
            email=user_data.email,
            password_hash= password_hash, 
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role,
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
        "email": new_user.email
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

def send_reset_email(to_email: str, token: str):
    reset_link = f"http://localhost:8000/reset-password?token={token}"
    subject = "Password Reset Request"
    body = f"Hi,\n\nClick the following link to reset your password:\n\n{reset_link}\n\nThis link expires in 1 hour.\n\nIf you didn't request this, ignore this email."

    # Construct the email
    message = MIMEMultipart()
    message["From"] = FROM_EMAIL
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
        print(f"Reset email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


#forgot password api
@app.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = RESET_PASSWORD_SECRET.dumps(user.email, salt="reset-password")
    background_tasks.add_task(send_reset_email, user.email, token)
    return {"message": "Reset link sent to your email"}


# reset password api
@app.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    try:
        email = RESET_PASSWORD_SECRET.loads(request.token, salt="reset-password", max_age=TOKEN_EXPIRATION_SECONDS)
    except SignatureExpired:
        raise HTTPException(status_code=400, detail="Token has expired")
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = pwd_context.hash(request.new_password)
    db.commit()

    return {"message": "Password has been reset successfully"}


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


def get_prediction_report(prediction_id: int, db: Session = Depends(get_db), format_type: str = "pdf"):
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

    # Create HTML content using the modern template
    html_content = generate_modern_report_html(patient, prediction)
    
    # If HTML format is requested, return HTML response
    if format_type.lower() == "html":
        return HTMLResponse(content=html_content, status_code=200)
    
    # Otherwise generate PDF using ReportLab
    os.makedirs("reports", exist_ok=True)
    filename = f"prediction_report_{prediction_id}.pdf"
    filepath = os.path.join("reports", filename)
    
    # Generate PDF using ReportLab
    generate_pdf_report(filepath, patient, prediction)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/pdf'
    )


def generate_pdf_report(filepath, patient, prediction):
    """Generate PDF report using ReportLab instead of WeasyPrint"""
    
    # Create a PDF document
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for the 'flowable' elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        textColor=colors.HexColor('#1d3557'),
        spaceAfter=12
    )
    
    info_label_style = ParagraphStyle(
        'InfoLabel',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    
    info_value_style = ParagraphStyle(
        'InfoValue',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#1d3557')
    )
    
    # Add title and header
    elements.append(Paragraph("Heart Disease Prediction Report", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Patient Information section
    elements.append(Paragraph("Patient Information", section_title_style))
    
    # Create patient info table
    patient_data = [
        ["Name", patient.full_name],
        ["Age", str(patient.age)],
        ["Gender", patient.gender],
        ["Email", patient.email]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecef'))
    ]))
    
    elements.append(patient_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Prediction Details section
    elements.append(Paragraph("Prediction Details", section_title_style))
    
    # Format date for display
    created_at = prediction.created_at.strftime("%Y-%m-%d %H:%M:%S") if prediction.created_at else "N/A"
    
    # Format confidence score for display
    confidence_score = f"{prediction.confidence_score:.2f}%" if prediction.confidence_score is not None else "N/A"
    
    # Create prediction info table
    prediction_data = [
        ["Prediction ID", str(prediction.prediction_id)],
        ["Model Used", prediction.model_used],
        ["Created At", created_at],
        ["Prediction", prediction.prediction_label],
        ["Risk Level", prediction.risk_level],
        ["Confidence Score", confidence_score]
    ]
    
    # Set colors based on risk level
    risk_color = colors.green
    if prediction.risk_level == "Medium":
        risk_color = colors.orange
    elif prediction.risk_level == "High":
        risk_color = colors.red
    
    prediction_table = Table(prediction_data, colWidths=[2*inch, 3*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecef')),
        ('BACKGROUND', (1, 4), (1, 4), risk_color),  # Risk level row
        ('TEXTCOLOR', (1, 4), (1, 4), colors.white)  # Risk level text color
    ]))
    
    elements.append(prediction_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Recommendations section
    elements.append(Paragraph("Recommendations for a Healthy Heart", section_title_style))
    
    # Recommendation 1
    elements.append(Paragraph("<b>Don't smoke or use tobacco</b>", normal_style))
    elements.append(Paragraph("Smoking increases your risk of heart disease and stroke by 2-4 times.", 
                            info_label_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Recommendation 2
    elements.append(Paragraph("<b>Get moving regularly</b>", normal_style))
    elements.append(Paragraph("Aim for at least 30 to 60 minutes of activity daily.", 
                            info_label_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Recommendation 3
    elements.append(Paragraph("<b>Eat a heart-healthy diet including:</b>", normal_style))
    
    diet_items = [
        "• Vegetables and fruits",
        "• Beans or other legumes",
        "• Lean meats and fish",
        "• Low-fat or fat-free dairy foods",
        "• Whole grains",
        "• Healthy fats such as olive oil and avocado"
    ]
    
    for item in diet_items:
        elements.append(Paragraph(item, info_label_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Footer
    elements.append(Paragraph(
        "<i>Note: Consult a cardiologist for treatment and diagnosis. ML Systems can make mistakes.</i>",
        info_label_style
    ))
    
    # Build the PDF
    doc.build(elements)


def generate_modern_report_html(patient, prediction):
    """Generate modern HTML report with the new stylish design"""
    
    # Format confidence score for display
    confidence_score = f"{prediction.confidence_score:.2f}%" if prediction.confidence_score is not None else "N/A"
    
    # Determine risk badge class based on risk level
    risk_badge_class = "low"
    if prediction.risk_level == "Medium":
        risk_badge_class = "medium"
    elif prediction.risk_level == "High":
        risk_badge_class = "high"
    
    # Format date for display
    created_at = prediction.created_at.strftime("%Y-%m-%d %H:%M:%S") if prediction.created_at else "N/A"
    
    # Create the HTML content with the modern design
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction Report</title>
    <style>
        :root {{
            --primary: #e63946;
            --secondary: #457b9d;
            --light: #f1faee;
            --dark: #1d3557;
            --accent: #a8dadc;
        }}
        
        body, html {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }}
        
        .container {{
            max-width: 800px;
            margin: 2rem auto;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary), #ff6b6b);
            color: white;
            padding: 2rem;
            text-align: center;
            position: relative;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 600;
        }}
        
        .header svg {{
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 3rem;
        }}
        
        .card {{
            background: white;
            padding: 2rem;
            margin-bottom: 1px;
        }}
        
        .section-title {{
            color: var(--dark);
            font-size: 1.4rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}
        
        .section-title i {{
            color: var(--primary);
            font-size: 1.5rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}
        
        .info-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .info-label {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 0.3rem;
        }}
        
        .info-value {{
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--dark);
        }}
        
        .prediction-box {{
            background-color: #f8f9fa;
            border-left: 4px solid var(--secondary);
            padding: 1.5rem;
            border-radius: 6px;
            margin-top: 1rem;
        }}
        
        .prediction-result {{
            display: flex;
            align-items: center;
            margin-top: 1rem;
        }}
        
        .risk-badge {{
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            color: white;
            background-color: #28a745; /* Green for Low risk */
            margin-left: 1rem;
        }}
        
        .risk-badge.medium {{
            background-color: #ffc107; /* Yellow for Medium risk */
        }}
        
        .risk-badge.high {{
            background-color: #dc3545; /* Red for High risk */
        }}
        
        .confidence-bar {{
            margin-top: 1.5rem;
            background-color: #e9ecef;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }}
        
        .confidence-progress {{
            height: 100%;
            background-color: var(--secondary);
            width: {prediction.confidence_score}%; /* Set dynamically based on confidence score */
        }}
        
        .recommendations {{
            margin-top: 1.5rem;
        }}
        
        .recommendation-list {{
            list-style: none;
            padding: 0;
        }}
        
        .recommendation-item {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 1.2rem;
            padding-bottom: 1.2rem;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .recommendation-item:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
        
        .recommendation-icon {{
            background-color: var(--accent);
            color: var(--dark);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            flex-shrink: 0;
        }}
        
        .diet-list {{
            list-style: none;
            padding: 0;
            margin: 0.5rem 0 0 3.2rem;
        }}
        
        .diet-item {{
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
        }}
        
        .diet-item i {{
            color: var(--secondary);
            margin-right: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .footer {{
            background-color: #f8f9fa;
            padding: 1.5rem 2rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            font-style: italic;
            border-top: 1px solid #e9ecef;
        }}
        
        @media print {{
            body, html {{
                background-color: #fff;
            }}
            .container {{
                margin: 0;
                box-shadow: none;
            }}
        }}
    </style>
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-heartbeat"></i> Heart Disease Prediction Report</h1>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
                <path fill="#ffffff" fill-opacity="1" d="M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,128C672,128,768,160,864,160C960,160,1056,128,1152,112C1248,96,1344,96,1392,96L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
            </svg>
        </div>
        
        <div class="card">
            <h2 class="section-title">
                <i class="fas fa-user-circle"></i> Patient Information
            </h2>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Name</span>
                    <span class="info-value">{patient.full_name}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Age</span>
                    <span class="info-value">{patient.age}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Gender</span>
                    <span class="info-value">{patient.gender}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Email</span>
                    <span class="info-value">{patient.email}</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 class="section-title">
                <i class="fas fa-chart-line"></i> Prediction Details
            </h2>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Prediction ID</span>
                    <span class="info-value">{prediction.prediction_id}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Model Used</span>
                    <span class="info-value">{prediction.model_used}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Created At</span>
                    <span class="info-value">{created_at}</span>
                </div>
            </div>
            
            <div class="prediction-box">
                <div class="prediction-result">
                    <div>
                        <span class="info-label">Prediction</span>
                        <span class="info-value" style="font-size: 1.3rem; font-weight: 600;">{prediction.prediction_label}</span>
                    </div>
                    <span class="risk-badge {risk_badge_class}">{prediction.risk_level} Risk</span>
                </div>
                
                <div style="margin-top: 1.5rem;">
                    <span class="info-label">Confidence Score</span>
                    <div class="confidence-bar">
                        <div class="confidence-progress"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8rem; color: #6c757d;">
                        <span>0%</span>
                        <span>{confidence_score}</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 class="section-title">
                <i class="fas fa-heart"></i> Recommendations for a Healthy Heart
            </h2>
            <div class="recommendations">
                <ul class="recommendation-list">
                    <li class="recommendation-item">
                        <div class="recommendation-icon">
                            <i class="fas fa-smoking-ban"></i>
                        </div>
                        <div>
                            <strong>Don't smoke or use tobacco.</strong>
                            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Smoking increases your risk of heart disease and stroke by 2-4 times.</p>
                        </div>
                    </li>
                    <li class="recommendation-item">
                        <div class="recommendation-icon">
                            <i class="fas fa-running"></i>
                        </div>
                        <div>
                            <strong>Get moving regularly.</strong>
                            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Aim for at least 30 to 60 minutes of activity daily.</p>
                        </div>
                    </li>
                    <li class="recommendation-item">
                        <div class="recommendation-icon">
                            <i class="fas fa-apple-alt"></i>
                        </div>
                        <div>
                            <strong>Eat a heart-healthy diet including:</strong>
                            <ul class="diet-list">
                                <li class="diet-item"><i class="fas fa-carrot"></i> Vegetables and fruits</li>
                                <li class="diet-item"><i class="fas fa-seedling"></i> Beans or other legumes</li>
                                <li class="diet-item"><i class="fas fa-fish"></i> Lean meats and fish</li>
                                <li class="diet-item"><i class="fas fa-cheese"></i> Low-fat or fat-free dairy foods</li>
                                <li class="diet-item"><i class="fas fa-bread-slice"></i> Whole grains</li>
                                <li class="diet-item"><i class="fas fa-oil-can"></i> Healthy fats such as olive oil and avocado</li>
                            </ul>
                        </div>
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            Note: Consult a cardiologist for treatment and diagnosis. ML Systems can make mistakes.
        </div>
    </div>
</body>
</html>
    """
    
    return html_content


# Updated API endpoint with option to choose format
@app.get("/prediction/report/{prediction_id}")
def get_prediction_report_endpoint(prediction_id: int, format_type: str = "pdf", db: Session = Depends(get_db)):
    return get_prediction_report(prediction_id, db, format_type)
