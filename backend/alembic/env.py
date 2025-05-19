print("DEBUG: alembic/env.py execution started!", flush=True)

# Standard library imports
import os
import sys
from logging.config import fileConfig

# Third party imports
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Local application imports
# Add the project root (parent directory of 'alembic') to the Python path
# This allows Alembic to find 'app.core.config', 'app.models.base', etc.

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)



from app.core.config import settings # Import application settings
from app.models.base import Base     # Import Base from where all models inherit

# Import all your models so Base.metadata knows about them
# and Alembic can detect changes for autogeneration.

from app.models.user import User
from app.models.agent import AgentConfiguration # add all model classes
from app.models.job import JobHistory
from app.models.log import TaskLog

print(f"DEBUG: Tables registered in Base.metadata BEFORE target_metadata assignment: {list(Base.metadata.tables.keys())}", flush=True)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name) if config.config_file_name else None

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

print(f"DEBUG: Tables in target_metadata AFTER assignment: {list(target_metadata.tables.keys())}", flush=True)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    # Create a configuration dictionary for engine_from_config
# This ensures that the URL is correctly interpreted by SQLAlchemy
    
    db_config = config.get_section(config.config_ini_section, {})
    print(f"DEBUG: DATABASE_URL from settings in env.py is: {settings.DATABASE_URL!r}")
    db_config["sqlalchemy.url"] = settings.alembic_url
    connectable = engine_from_config(
        db_config,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type= True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
