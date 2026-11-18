"""Account and product persistence. Use DATABASE_URL for PostgreSQL in production."""
import os
import time
import uuid
from pathlib import Path
from sqlalchemy import create_engine, String, Text, Boolean, Float, ForeignKey, JSON, Integer, UniqueConstraint, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

class Base(DeclarativeBase):
    pass

def uid(): return str(uuid.uuid4())

class User(Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(Text)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(10), default='USER')
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_login: Mapped[float | None] = mapped_column(Float, nullable=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)

class Session(Base):
    __tablename__ = 'user_sessions'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    expires_at: Mapped[float] = mapped_column(Float)

class PasswordReset(Base):
    __tablename__ = 'password_resets'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    expires_at: Mapped[float] = mapped_column(Float)

class Analysis(Base):
    __tablename__ = 'analyses'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    repository: Mapped[str] = mapped_column(Text)
    pull_request: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)
    status: Mapped[str] = mapped_column(String(30), default='PENDING')
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    events: Mapped[list] = mapped_column(JSON, default=list)
    repository_connection_id: Mapped[str | None] = mapped_column(ForeignKey('repository_connections.id'), nullable=True, index=True)
    pull_request_id: Mapped[str | None] = mapped_column(ForeignKey('pull_requests.id'), nullable=True, index=True)

class RepositoryConnection(Base):
    __tablename__ = 'repository_connections'
    __table_args__ = (UniqueConstraint('user_id', 'provider', 'repository_id', name='uq_user_provider_repository'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    repository: Mapped[str] = mapped_column(Text)  # legacy alias of full_name
    provider: Mapped[str] = mapped_column(String(30), default='manual', index=True)
    installation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    repository_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    repo_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    connection_status: Mapped[str] = mapped_column(String(30), default='CONNECTED')
    connected_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_synced_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_mode: Mapped[str] = mapped_column(String(20), default='manual')

class PullRequest(Base):
    __tablename__ = 'pull_requests'
    __table_args__ = (UniqueConstraint('repository_connection_id', 'provider_pr_id', name='uq_connection_provider_pr'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    repository_connection_id: Mapped[str] = mapped_column(ForeignKey('repository_connections.id'), index=True)
    provider_pr_id: Mapped[str] = mapped_column(String(64))
    pr_number: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(30), index=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    base_sha: Mapped[str] = mapped_column(String(64))
    head_sha: Mapped[str] = mapped_column(String(64))
    base_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    head_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    html_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    opened_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    merged_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_synced_at: Mapped[float] = mapped_column(Float, default=time.time)

class GitHubInstallationIntent(Base):
    __tablename__ = 'github_installation_intents'
    state_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    expires_at: Mapped[float] = mapped_column(Float)

class AuditEvent(Base):
    __tablename__ = 'audit_events'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    actor_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    action: Mapped[str] = mapped_column(String(80))
    subject_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _database_url() -> str:
    """Resolve local SQLite storage from the project root, never the process CWD."""
    configured = os.getenv('DATABASE_URL', 'sqlite:///./backend/accounts.sqlite3')
    parsed = make_url(configured)
    if parsed.drivername.startswith('sqlite') and parsed.database not in (None, '', ':memory:'):
        database = Path(parsed.database)
        if not database.is_absolute():
            database = (PROJECT_ROOT / database).resolve()
        parsed = parsed.set(database=str(database))
    return parsed.render_as_string(hide_password=False)


url = _database_url()
engine = create_engine(url, connect_args={'check_same_thread': False} if url.startswith('sqlite') else {}, pool_pre_ping=True)
DB = sessionmaker(engine, expire_on_commit=False)

def initialize():
    if url.startswith('sqlite'):
        Base.metadata.create_all(engine)
        # create_all does not extend existing SQLite tables. Keep local installs
        # forward-compatible while production deployments use Alembic.
        columns = {column['name'] for column in inspect(engine).get_columns('analyses')}
        audit_columns = {column['name'] for column in inspect(engine).get_columns('audit_events')}
        repository_columns = {column['name'] for column in inspect(engine).get_columns('repository_connections')}
        with engine.begin() as connection:
            # Protect the evidence even from an already-running stale process.
            frozen_fields = ('finalized_demo', 'purpose', 'source_analysis_id', 'artifact_origin',
                             'provider_request_count', 'replayed', 'analysis', 'validation', 'repair',
                             'test_generation', 'candidate_comparison', 'repair_attempts', 'ci_pipeline',
                             'pre_repair_decision', 'post_candidate_decision', 'decision')
            differences = ' OR '.join(
                f"json_extract(NEW.result, '$.{key}') IS NOT json_extract(OLD.result, '$.{key}')"
                for key in frozen_fields
            )
            connection.execute(text(f"""
                CREATE TRIGGER IF NOT EXISTS protect_final_demo_evidence
                BEFORE UPDATE ON analyses
                WHEN json_extract(OLD.result, '$.finalized_demo') = 1
                  AND (NEW.status IS NOT OLD.status OR {differences})
                BEGIN SELECT RAISE(ABORT, 'Final demo evidence is immutable'); END
            """))
            if 'updated_at' not in columns:
                connection.execute(text("ALTER TABLE analyses ADD COLUMN updated_at FLOAT"))
                connection.execute(text("UPDATE analyses SET updated_at = created_at WHERE updated_at IS NULL"))
            if 'details' not in audit_columns:
                connection.execute(text("ALTER TABLE audit_events ADD COLUMN details JSON"))
                connection.execute(text("UPDATE audit_events SET details = '{}' WHERE details IS NULL"))
            for name, ddl in {
                'repository_connection_id': 'VARCHAR(36)', 'pull_request_id': 'VARCHAR(36)'
            }.items():
                if name not in columns: connection.execute(text(f'ALTER TABLE analyses ADD COLUMN {name} {ddl}'))
            repository_additions = {
                'installation_id':'VARCHAR(64)', 'repository_id':'VARCHAR(64)', 'owner':'VARCHAR(255)',
                'repo_name':'VARCHAR(255)', 'full_name':'TEXT', 'default_branch':'VARCHAR(255)',
                'connection_status':"VARCHAR(30) DEFAULT 'CONNECTED'", 'connected_at':'FLOAT',
                'last_synced_at':'FLOAT', 'analysis_mode':"VARCHAR(20) DEFAULT 'manual'",
            }
            for name, ddl in repository_additions.items():
                if name not in repository_columns: connection.execute(text(f'ALTER TABLE repository_connections ADD COLUMN {name} {ddl}'))

def public_user(user):
    return {k: getattr(user, k) for k in ('id', 'email', 'display_name', 'avatar_url', 'role', 'is_active', 'created_at', 'updated_at', 'last_login', 'preferences')}
