"""Add index to task_logs.job_id

Revision ID: 47690fe77083
Revises: 0975e8cef8bf
Create Date: 2025-05-19 01:32:59.276188

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '47690fe77083'
down_revision: Union[str, None] = '0975e8cef8bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(op.f('ix_task_logs_job_id'), 'task_logs', ['job_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_task_logs_job_id'), table_name='task_logs')
    # ### end Alembic commands ###
