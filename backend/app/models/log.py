# app/models/log.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base # Import our Base class

class TaskLog(Base):
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("job_history.id"), nullable=False)
    # user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Optional, can be inferred via job_id
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    log_level = Column(String, default="INFO") # e.g., INFO, ERROR, DEBUG, TOOL_CALL
    message = Column(Text, nullable=True)
    details = Column(JSON, nullable=True) # For structured log data like tool args/results

    # Relationship
    # job = relationship("JobHistory", back_populates="task_logs")

    def __repr__(self):
        return f"<TaskLog(id={self.id}, job_id={self.job_id}, level=\'{self.log_level}\')>"
