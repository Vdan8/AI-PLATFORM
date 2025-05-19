# backend/app/api/job.py
from fastapi import APIRouter, Depends, status, HTTPException # Import HTTPException
from app.models.user import User
from app.core.auth import get_current_user
from app.utils.logger import logger
from app.models.base import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.job import JobHistory  # Import your JobHistory model
from app.exceptions import JobNotFoundError, UnauthorizedToAccessJobError

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_job(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Endpoint to create a new job."""
    try:
        # 1.  Create a new JobHistory object.  You'll need a JobHistory model.
        job = JobHistory(
            user_id=user.id,
            status="pending",  #  initial status
            # Add other relevant fields like agent_config_id, input, etc.
        )
        db.add(job)

        # 2. Commit the changes to the database.
        await db.commit()

        # 3. Refresh the job object to get the generated ID.
        await db.refresh(job)

        # 4.  Return the created job.
        return job
    except Exception as e:
        logger.error(f"Failed to create job for user {user.id}: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job",
        )


@router.get("/{job_id}")
async def get_job(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to retrieve details of a specific job."""
    try:
        # 1. Construct the query to fetch the job, ensuring the user has access.
        result = await db.execute(
            select(JobHistory).where(
                JobHistory.id == job_id, JobHistory.user_id == user.id
            )
        )
        job = result.scalar_one_or_none()

        # 2.  Raise an exception if the job is not found or the user doesn't have access.
        if not job:
            raise UnauthorizedToAccessJobError(job_id=job_id)

        # 3.  Return the job details.
        return job
    except (JobNotFoundError, UnauthorizedToAccessJobError) as e:
        await db.rollback()
        raise e
    except Exception as e:
        logger.error(
            f"Failed to retrieve job {job_id} for user {user.id}: {str(e)}",
            exc_info=True,
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job {job_id}",
        )