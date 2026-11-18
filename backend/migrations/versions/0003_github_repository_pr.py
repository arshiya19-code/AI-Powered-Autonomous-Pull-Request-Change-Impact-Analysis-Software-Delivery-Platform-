"""GitHub App repository and pull request persistence.

Revision ID: 0003_github_repository_pr
Revises: 0002_admin_activity
"""
from alembic import op
import sqlalchemy as sa

revision = '0003_github_repository_pr'
down_revision = '0002_admin_activity'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    repository_columns = {column['name'] for column in inspector.get_columns('repository_connections')}
    additions = {
        'installation_id': sa.Column('installation_id', sa.String(64), nullable=True),
        'repository_id': sa.Column('repository_id', sa.String(64), nullable=True),
        'owner': sa.Column('owner', sa.String(255), nullable=True),
        'repo_name': sa.Column('repo_name', sa.String(255), nullable=True),
        'full_name': sa.Column('full_name', sa.Text(), nullable=True),
        'default_branch': sa.Column('default_branch', sa.String(255), nullable=True),
        'connection_status': sa.Column('connection_status', sa.String(30), nullable=False, server_default='CONNECTED'),
        'connected_at': sa.Column('connected_at', sa.Float(), nullable=True),
        'last_synced_at': sa.Column('last_synced_at', sa.Float(), nullable=True),
        'analysis_mode': sa.Column('analysis_mode', sa.String(20), nullable=False, server_default='manual'),
    }
    missing = [column for name, column in additions.items() if name not in repository_columns]
    if missing:
        with op.batch_alter_table('repository_connections') as batch:
            for column in missing:
                batch.add_column(column)

    inspector = sa.inspect(bind)
    repository_indexes = {index['name'] for index in inspector.get_indexes('repository_connections')}
    repository_uniques = {constraint['name'] for constraint in inspector.get_unique_constraints('repository_connections')}
    with op.batch_alter_table('repository_connections') as batch:
        if 'ix_repository_connections_installation_id' not in repository_indexes:
            batch.create_index('ix_repository_connections_installation_id', ['installation_id'])
        if 'ix_repository_connections_repository_id' not in repository_indexes:
            batch.create_index('ix_repository_connections_repository_id', ['repository_id'])
        if 'uq_user_provider_repository' not in repository_uniques:
            batch.create_unique_constraint('uq_user_provider_repository', ['user_id','provider','repository_id'])

    if 'pull_requests' not in tables:
        op.create_table('pull_requests',
            sa.Column('id', sa.String(36), primary_key=True),
            sa.Column('repository_connection_id', sa.String(36), sa.ForeignKey('repository_connections.id'), nullable=False),
            sa.Column('provider_pr_id', sa.String(64), nullable=False), sa.Column('pr_number', sa.Integer(), nullable=False),
            sa.Column('title', sa.Text(), nullable=False), sa.Column('state', sa.String(30), nullable=False),
            sa.Column('author', sa.String(255)), sa.Column('base_sha', sa.String(64), nullable=False),
            sa.Column('head_sha', sa.String(64), nullable=False), sa.Column('base_branch', sa.String(255)),
            sa.Column('head_branch', sa.String(255)), sa.Column('html_url', sa.Text()),
            sa.Column('opened_at', sa.Float()), sa.Column('updated_at', sa.Float()), sa.Column('merged_at', sa.Float()),
            sa.Column('last_synced_at', sa.Float(), nullable=False),
            sa.UniqueConstraint('repository_connection_id','provider_pr_id',name='uq_connection_provider_pr'))
    inspector = sa.inspect(bind)
    pull_indexes = {index['name'] for index in inspector.get_indexes('pull_requests')}
    for name, columns in {
        'ix_pull_requests_repository_connection_id':['repository_connection_id'],
        'ix_pull_requests_pr_number':['pr_number'],
        'ix_pull_requests_state':['state'],
    }.items():
        if name not in pull_indexes:
            op.create_index(name, 'pull_requests', columns)

    if 'github_installation_intents' not in tables:
        op.create_table('github_installation_intents', sa.Column('state_hash',sa.String(64),primary_key=True), sa.Column('user_id',sa.String(36),sa.ForeignKey('users.id'),nullable=False), sa.Column('expires_at',sa.Float(),nullable=False))
    inspector = sa.inspect(bind)
    intent_indexes = {index['name'] for index in inspector.get_indexes('github_installation_intents')}
    if 'ix_github_installation_intents_user_id' not in intent_indexes:
        op.create_index('ix_github_installation_intents_user_id','github_installation_intents',['user_id'])

    analysis_columns = {column['name'] for column in inspector.get_columns('analyses')}
    with op.batch_alter_table('analyses') as batch:
        if 'repository_connection_id' not in analysis_columns:
            batch.add_column(sa.Column('repository_connection_id',sa.String(36),sa.ForeignKey('repository_connections.id')))
        if 'pull_request_id' not in analysis_columns:
            batch.add_column(sa.Column('pull_request_id',sa.String(36),sa.ForeignKey('pull_requests.id')))
    inspector = sa.inspect(bind)
    analysis_indexes = {index['name'] for index in inspector.get_indexes('analyses')}
    with op.batch_alter_table('analyses') as batch:
        if 'ix_analyses_repository_connection_id' not in analysis_indexes:
            batch.create_index('ix_analyses_repository_connection_id',['repository_connection_id'])
        if 'ix_analyses_pull_request_id' not in analysis_indexes:
            batch.create_index('ix_analyses_pull_request_id',['pull_request_id'])

def downgrade():
    with op.batch_alter_table('analyses') as batch:
        batch.drop_column('pull_request_id'); batch.drop_column('repository_connection_id')
    op.drop_table('github_installation_intents'); op.drop_table('pull_requests')
    with op.batch_alter_table('repository_connections') as batch:
        for name in ('analysis_mode','last_synced_at','connected_at','connection_status','default_branch','full_name','repo_name','owner','repository_id','installation_id'): batch.drop_column(name)
