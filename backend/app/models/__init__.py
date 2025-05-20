# backend/app/models/__init__.py
# Import all models here so Alembic can discover them
from app.models.base import Base
from app.models.user import User
from app.models.agent import AgentConfiguration
from app.models.token import RefreshToken
from app.models.log import TaskLog
from app.models.job import JobHistory
from app.models.tool import Tool # ADD THIS LINE