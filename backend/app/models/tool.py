# backend/app/models/tool.py
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base # Assuming your Base is in app.models.base

class Tool(Base):
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=False)
    parameters = Column(JSON, nullable=False, default={}) # Stores the schema as JSONB
    code = Column(Text, nullable=False) # Stores the Python code of the tool
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Optional: If you want to link tools to users or organizations
    # owner_id = Column(Integer, ForeignKey("users.id"))
    # owner = relationship("User", back_populates="tools")

    def __repr__(self):
        return f"<Tool(id={self.id}, name='{self.name}')>"