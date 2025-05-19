from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.config import settings
from passlib.context import CryptContext
import uuid
from app.models.base import get_db  # Import get_db here

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto"
)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token() -> str:
    return str(uuid.uuid4())

def get_user_by_email(email: str) -> User | None:
    """Fetch user by email."""
    db: Session = next(get_db())  # Get a database session
    try:
        user = db.query(User).filter(User.email == email).first()
        return user
    finally:
        db.close()  # Ensure the session is closed

async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> User:
    """Validate JWT and return authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exception
        # Expiration is automatically checked by jwt.decode
    except JWTError:
        raise credentials_exception

    user = get_user_by_email(email) # Call get_user_by_email without db
    if not user:
        raise credentials_exception
    return user