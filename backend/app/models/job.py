# app/models/job.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base # Import our Base class

class JobHistory(Base):
    __tablename__ = "job_history"

    id = Column(Integer, primary_key=True, index=True) # Could also be a UUID if preferred
    celery_task_id = Column(String, unique=True, index=True, nullable=True) # If using Celery
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_config_id = Column(Integer, ForeignKey("agent_configurations.id"), nullable=True)
    
    goal = Column(Text, nullable=False)
    status = Column(String, default="PENDING", index=True) # e.g., PENDING, RUNNING, COMPLETED, FAILED
    result = Column(Text, nullable=True) # Store final result or error message
    
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # user = relationship("User") # Add back_populates in User model later
    # agent_config = relationship("AgentConfiguration", back_populates="job_history")
    # task_logs = relationship("TaskLog", back_populates="job")

    def __repr__(self):
        return f"<JobHistory(id={self.id}, status=\'{self.status}\')>"
