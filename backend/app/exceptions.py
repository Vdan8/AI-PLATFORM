# backend/app/exceptions.py
from fastapi import HTTPException, status

class CustomException(HTTPException):
    """Base class for custom exceptions."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

class UserAlreadyExistsError(CustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

class IncorrectCredentialsError(CustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

class InvalidRefreshTokenError(CustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

class ExpiredRefreshTokenError(CustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )

class AgentNameAlreadyExistsError(CustomException):
    def __init__(self, agent_name: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with name '{agent_name}' already exists for this user.",
        )

class JobNotFoundError(CustomException):
    def __init__(self, job_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

class UnauthorizedToAccessJobError(CustomException):
    def __init__(self, job_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,  # Using 404 for consistency
            detail=f"Job {job_id} not found or user not authorized",
        )