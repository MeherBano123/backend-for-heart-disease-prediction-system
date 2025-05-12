## apis for user model

from fastapi import FastAPI,HTTPException, Depends,Request,status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import SQLModel, Field, Session, create_engine , select
from typing import Optional
from typing import Optional, List
from datetime import datetime , date,timedelta
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from .models import *
from .schemas import *
import bcrypt
from .main import *
from fastapi.responses import JSONResponse

def get_session():
    with Session(engine) as session:
        yield session


        
# main.py
from fastapi import FastAPI,HTTPException, Depends,Request,status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import SQLModel, Field, Session, create_engine , select
from typing import Optional
from typing import Optional, List
from datetime import datetime , date,timedelta
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from .models import *
from .schemas import *
import bcrypt
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from .users import *

app = FastAPI()

# Database engine setup
engine = create_engine("sqlite:///db.sqlite") 


# Create tables when application starts
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Security configurations
## change secret key later
SECRET_KEY = "your-secret-key"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 45

def get_session():
    with Session(engine) as session:
        yield session


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

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_current_user(token: str = Depends(oauth2_scheme),
                         session: Session = Depends(get_session)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account is inactive"
        )
    return current_user

async def get_current_active_admin(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user 

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str




# Function to authenticate user
def authenticate_user(username: str, password: str, session: Session):
    user = session.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user


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

    
def get_current_user_id(token_data: TokenData = Depends(decode_token)) -> int:
    return token_data.user_id



# Functions for password handling
def hash_password(password: str) -> str:
    # Generate a salt and hash the password
    if not isinstance(password, str):
        raise ValueError("Password must be a string")
        
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


@app.post("/users/")
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
        db_user = User(
            
            username=user_data.username,
            email=user_data.email,
            password_hash= password_hash, 
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role,
            created_at=user_data.created_at,
            last_login=user_data.last_login
        )
        
        # Ensure timestamps are datetime objects
        if isinstance(db_user.created_at, str):
            db_user.created_at = datetime.fromisoformat(db_user.created_at.replace("Z", "+00:00"))
        if isinstance(db_user.last_login, str):
            db_user.last_login = datetime.fromisoformat(db_user.last_login.replace("Z", "+00:00"))
        
    
        # Print debug information
        print(f"Password hash: {password_hash}")
        print(f"User data: {db_user.__dict__}")
        
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
    except Exception as e:
        session.rollback()
        print(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@app.get("/users/", response_model=List[User])
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
    
    Args:
        data: Dictionary containing the claims (e.g., {"sub": "username"})
        expires_delta: Optional timedelta for token expiration
        secret_key: Secret key for signing the token
        algorithm: Encryption algorithm to use
        
    Returns:
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
    user = authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    session.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user.id),
        "username": user.username
    }
