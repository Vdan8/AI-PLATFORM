# app/api/auth.py
from fastapi import APIRouter, Depends, status, HTTPException # Import HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.schemas.auth import UserCreate, UserLogin, Token, RefreshToken
from app.models.base import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.logger import logger
from app.core.auth import create_access_token, create_refresh_token, settings
from app.models.user import User
from app.models.token import RefreshToken as RefreshTokenModel
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.exceptions import (  # Import custom exceptions
    UserAlreadyExistsError,
    IncorrectCredentialsError,
    InvalidRefreshTokenError,
    ExpiredRefreshTokenError,
)

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_user_by_email(email: str, db: AsyncSession):
    """Fetch user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def store_refresh_token(db: AsyncSession, user: User, token: str):
    """Store refresh token in database."""
    expire_date = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db_token = RefreshTokenModel(user_id=user.id, token=token, expires_at=expire_date)
    db.add(db_token)
    await db.commit()
    await db.refresh(db_token)  # Use await db.refresh
    return db_token


@router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    try:
        if await get_user_by_email(user_data.email, db):
            raise UserAlreadyExistsError()  # Use custom exception

        user = User(email=user_data.email)
        user.set_password(user_data.password)
        db.add(user)
        await db.commit()
        await db.refresh(user)

        access_token = create_access_token(data={"sub": user.email})
        refresh_token_value = create_refresh_token()
        await store_refresh_token(db, user, refresh_token_value)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token_value,
        }
    except UserAlreadyExistsError as e:  # Catch the custom exception
        await db.rollback()
        raise e
    except Exception as e:
        logger.error(
            f"Registration failed for user {user_data.email}: {str(e)}", exc_info=True
        )
        await db.rollback()
        raise HTTPException( # Raise HTTPException here
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Log in a user."""
    try:
        user = await get_user_by_email(user_data.email, db)
        if not user or not user.verify_password(user_data.password):
            raise IncorrectCredentialsError()  # Use custom exception

        access_token = create_access_token(data={"sub": user.email})
        refresh_token_value = create_refresh_token()
        await store_refresh_token(db, user, refresh_token_value)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token_value,
        }
    except IncorrectCredentialsError as e:  # Catch the custom exception
        raise e
    except Exception as e:
        logger.error(f"Login failed for user {user_data.email}: {str(e)}", exc_info=True)
        raise HTTPException( # Raise HTTPException here
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    refresh_token_data: RefreshToken, db: AsyncSession = Depends(get_db)
):
    """Refresh an access token using a refresh token."""
    try:
        refresh_token = refresh_token_data.refresh_token
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Refresh token required",
            )

        result = await db.execute(
            select(RefreshTokenModel)
            .where(RefreshTokenModel.token == refresh_token)
            .options(selectinload(RefreshTokenModel.user))  # Eager load the user
        )
        db_refresh_token = result.scalar_one_or_none()

        if not db_refresh_token:
            raise InvalidRefreshTokenError()  # Use custom exception

        user = db_refresh_token.user
        if not user:
            raise InvalidRefreshTokenError()  # Use custom exception

        if db_refresh_token.expires_at and datetime.utcnow() > db_refresh_token.expires_at:
            raise ExpiredRefreshTokenError()  # Use custom exception

        access_token = create_access_token(data={"sub": user.email})
        new_refresh_token_value = create_refresh_token()
        await store_refresh_token(db, user, new_refresh_token_value)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": new_refresh_token_value,
        }
    except (
        InvalidRefreshTokenError,
        ExpiredRefreshTokenError,
    ) as e:  # Catch custom exceptions
        await db.rollback()
        raise e
    except Exception as e:
        logger.error(f"Failed to refresh token: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException( # Raise HTTPException here
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )
