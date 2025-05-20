# app/models/agent.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base # Import our Base class

class AgentConfiguration(Base):
    __tablename__ = "agent_configurations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    system_prompt = Column(Text, nullable=False)
    # Store list of tool names or more complex tool configurations
    tools_config = Column(JSON, nullable=True) 
    llm_model_name = Column(String, nullable=True) # e.g., to specify a particular model for this agent
    max_steps = Column(Integer, default=10)
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="agents") # We'll uncomment/add this when User model is ready for it

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to JobHistory
    job_history = relationship("JobHistory", back_populates="agent_config")

    def __repr__(self):
        return f"<AgentConfiguration(id={self.id}, name=\'{self.name}\')>"
