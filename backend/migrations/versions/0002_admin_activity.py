"""Add timestamps and safe audit metadata used by platform administration."""
from alembic import op
import sqlalchemy as sa

revision = '0002_admin_activity'
down_revision = '0001_accounts'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    analysis_columns = {column['name'] for column in inspector.get_columns('analyses')}
    audit_columns = {column['name'] for column in inspector.get_columns('audit_events')}
    if 'updated_at' not in analysis_columns:
        with op.batch_alter_table('analyses') as batch:
            batch.add_column(sa.Column('updated_at', sa.Float(), nullable=True))
    op.execute('UPDATE analyses SET updated_at = created_at WHERE updated_at IS NULL')
    with op.batch_alter_table('analyses') as batch:
        batch.alter_column('updated_at', nullable=False)
    if 'details' not in audit_columns:
        with op.batch_alter_table('audit_events') as batch:
            batch.add_column(sa.Column('details', sa.JSON(), nullable=True))
    op.execute("UPDATE audit_events SET details = '{}' WHERE details IS NULL")
    with op.batch_alter_table('audit_events') as batch:
        batch.alter_column('details', nullable=False)

def downgrade():
    with op.batch_alter_table('audit_events') as batch:
        batch.drop_column('details')
    with op.batch_alter_table('analyses') as batch:
        batch.drop_column('updated_at')
