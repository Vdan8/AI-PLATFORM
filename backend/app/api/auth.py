# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.schemas.auth import UserCreate, UserLogin, Token
from app.models.base import get_db  # Now correctly imported
from sqlalchemy.orm import Session

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_user_by_email(email: str, db: Session):
    """Moved User import here to break circular dependency"""
    from app.models.user import User
    return db.query(User).filter(User.email == email).first()

@router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(user_data.email, db):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    from app.models.user import User
    from app.core.auth import create_access_token
    
    user = User(email=user_data.email)
    user.set_password(user_data.password)
    db.add(user)
    db.commit()
    
    return {
        "access_token": create_access_token(data={"sub": user.email}),
        "token_type": "bearer"
    }