# app/api/agent.py
from fastapi import APIRouter, Depends
from app.models.user import User
from app.core.auth import get_current_user

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/")
async def create_agent(user: User = Depends(get_current_user)):
    """Example protected endpoint"""
    return {
        "message": f"Agent created by {user.email}",
        "user_id": user.id
    }