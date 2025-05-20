# backend/app/crud/tool.py
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.tool import Tool # This import is correct
from app.schemas.tool import ToolCreate, ToolUpdate


ModelType = TypeVar("ModelType", bound=Tool)


class CRUDBase(Generic[ModelType]):
    """
    CRUD object with default methods to Create, Read, Update, Delete (CRUD).
    **Parameters**
    * `model`: A SQLAlchemy model class
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def get(self, db: AsyncSession, tool_id: int) -> Optional[ModelType]:
        """
        Retrieve a single object by its ID.
        """
        result = await db.execute(select(self.model).filter(self.model.id == tool_id))
        return result.scalar_one_or_none()

    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[ModelType]:
        """
        Retrieve a single object by its name.
        """
        result = await db.execute(select(self.model).filter(self.model.name == name))
        return result.scalar_one_or_none()

    async def get_all(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Retrieve multiple objects with pagination.
        """
        result = await db.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()

    async def create(self, db: AsyncSession, tool_in: ToolCreate) -> ModelType:
        """
        Create a new object.
        """
        obj_in_data = jsonable_encoder(tool_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(self, db: AsyncSession, db_tool: ModelType, tool_update: ToolUpdate) -> ModelType:
        """
        Update an existing object.
        """
        obj_data = jsonable_encoder(db_tool)
        update_data = tool_update.dict(exclude_unset=True)

        for field in obj_data:
            if field in update_data:
                setattr(db_tool, field, update_data[field])

        db.add(db_tool)
        await db.commit()
        await db.refresh(db_tool) # Changed db_obj to db_tool
        return db_tool

    async def delete(self, db: AsyncSession, tool_id: int) -> Optional[ModelType]:
        """
        Delete an object by its ID.
        """
        obj = await db.execute(select(self.model).filter(self.model.id == tool_id)).scalar_one_or_none()
        if obj:
            await db.delete(obj)
            await db.commit()
        return obj

# This is where tool_crud is actually defined and available for other modules to import.
tool_crud = CRUDBase[Tool](Tool)