# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from .base import Base  # Relative import now works
# from passlib.context import CryptContext # Removed, using auth.py
from sqlalchemy.orm import relationship

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # Removed, using auth.py

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # No need for pwd_context here, we'll use the functions from auth.py
    def set_password(self, password: str):
        from app.core.auth import get_password_hash
        self.hashed_password = get_password_hash(password)

    def verify_password(self, password: str) -> bool:
        from app.core.auth import verify_password
        return verify_password(password, self.hashed_password)

    # relationships
    agents = relationship("AgentConfiguration", back_populates="owner")
    jobs = relationship("JobHistory", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")