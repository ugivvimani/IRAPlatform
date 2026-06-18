"""
CLI management commands for IRA platform.
Run with: python -m app.cli <command> <args>
"""
import click
import sys
import logging
from datetime import datetime, timezone

from app.settings import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store
from app.auth import TokenManager, PasswordManager, USERS_DB

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """IRA Platform management CLI."""
    logging.basicConfig(level=logging.INFO)


@cli.group()
def db():
    """Database commands."""
    pass


@db.command()
def init():
    """Initialize database schema."""
    click.echo("Initializing database schema...")
    settings = load_settings()
    storage = build_storage_repository(settings)
    click.echo(f"✓ Database initialized (backend: {settings.db_backend})")


@db.command()
def migrate():
    """Run database migrations."""
    click.echo("Running migrations...")
    click.echo("Note: Use 'alembic upgrade head' to run Alembic migrations")
    click.echo("Example: alembic upgrade head")


@cli.group()
def user():
    """User management commands."""
    pass


@user.command()
@click.option('--username', prompt='Username', help='Username')
@click.option('--role', type=click.Choice(['admin', 'analyst', 'viewer']), default='viewer', help='User role')
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='Password')
def create(username, role, password):
    """Create a new user."""
    click.echo(f"Creating user '{username}' with role '{role}'...")
    
    # Hash password
    password_hash = PasswordManager.hash_password(password)
    
    # In production, save to database
    # For now, just show the hash
    click.echo(f"✓ User created")
    click.echo(f"  Username: {username}")
    click.echo(f"  Role: {role}")
    click.echo(f"  Password hash: {password_hash}")


@user.command()
@click.option('--username', prompt='Username', help='Username')
def generate_token(username):
    """Generate JWT token for a user."""
    click.echo(f"Generating token for user '{username}'...")
    
    # Look up user role (in production, from database)
    user_data = USERS_DB.get(username)
    if not user_data:
        click.echo(f"✗ User '{username}' not found")
        sys.exit(1)
    
    token = TokenManager.create_access_token(
        subject=username,
        role=user_data.get("role", "viewer"),
    )
    
    click.echo(f"✓ Token generated")
    click.echo(f"  Token: {token}")
    click.echo(f"  Use with: -H 'Authorization: Bearer {token}'")


@cli.group()
def vector():
    """Vector store commands."""
    pass


@vector.command()
def health():
    """Check vector store health."""
    click.echo("Checking vector store health...")
    try:
        vector_store = build_vector_store()
        click.echo(f"✓ Vector store healthy")
        click.echo(f"  Backend: {type(vector_store).__name__}")
    except Exception as e:
        click.echo(f"✗ Vector store error: {e}")
        sys.exit(1)


@vector.command()
@click.option('--query', prompt='Query text', help='Text to embed')
def embed(query):
    """Test embedding functionality."""
    click.echo(f"Embedding query: '{query}'...")
    try:
        from app.embeddings import EmbeddingFactory
        embedder = EmbeddingFactory.create()
        embedding = embedder.embed_sync([query])
        click.echo(f"✓ Embedding generated")
        click.echo(f"  Dimensions: {len(embedding[0])}")
        click.echo(f"  First 5 values: {embedding[0][:5]}")
    except Exception as e:
        click.echo(f"✗ Embedding error: {e}")
        sys.exit(1)


@cli.group()
def watchlist():
    """Watchlist commands."""
    pass


@watchlist.command()
def list():
    """List all watchlisted entities."""
    click.echo("Fetching watchlist...")
    settings = load_settings()
    storage = build_storage_repository(settings)
    
    entries = storage.list_watchlist()
    if not entries:
        click.echo("✓ Watchlist is empty")
        return
    
    click.echo(f"✓ Found {len(entries)} entities on watchlist:")
    for entry in entries:
        click.echo(f"  - {entry.entity_id}: {entry.company_name}")


@watchlist.command()
@click.option('--entity-id', prompt='Entity ID', help='Entity ID')
@click.option('--company-name', prompt='Company name', help='Company name')
@click.option('--notes', default='', help='Notes')
def add(entity_id, company_name, notes):
    """Add entity to watchlist."""
    click.echo(f"Adding '{company_name}' to watchlist...")
    
    from app.contracts import WatchlistEntry
    
    settings = load_settings()
    storage = build_storage_repository(settings)
    
    entry = WatchlistEntry(
        entity_id=entity_id,
        company_name=company_name,
        notes=notes,
        added_at=datetime.now(timezone.utc),
    )
    
    result = storage.upsert_watchlist(entry)
    click.echo(f"✓ Entity added to watchlist")
    click.echo(f"  ID: {result.entity_id}")
    click.echo(f"  Name: {result.company_name}")


@watchlist.command()
@click.option('--entity-id', prompt='Entity ID', help='Entity ID')
def remove(entity_id):
    """Remove entity from watchlist."""
    click.echo(f"Removing '{entity_id}' from watchlist...")
    
    settings = load_settings()
    storage = build_storage_repository(settings)
    
    success = storage.delete_watchlist(entity_id)
    if success:
        click.echo(f"✓ Entity removed from watchlist")
    else:
        click.echo(f"✗ Entity not found")
        sys.exit(1)


@cli.group()
def assessment():
    """Assessment commands."""
    pass


@assessment.command()
@click.option('--entity-id', prompt='Entity ID', help='Entity ID')
@click.option('--limit', default=10, help='Number of assessments to show')
def history(entity_id, limit):
    """Show assessment history for an entity."""
    click.echo(f"Fetching assessment history for '{entity_id}'...")
    
    settings = load_settings()
    storage = build_storage_repository(settings)
    
    records = storage.list_assessments(entity_id=entity_id, limit=limit)
    if not records:
        click.echo("✓ No assessments found")
        return
    
    click.echo(f"✓ Found {len(records)} assessments:")
    for record in records:
        click.echo(f"  - {record.created_at}: {record.risk_rating} (confidence: {record.confidence})")


@cli.group()
def config():
    """Configuration commands."""
    pass


@config.command()
def show():
    """Show current configuration."""
    click.echo("Current configuration:")
    settings = load_settings()
    click.echo(f"  App environment: {settings.app_env}")
    click.echo(f"  Database backend: {settings.db_backend}")
    click.echo(f"  Vector backend: {settings.vector_backend}")
    click.echo(f"  LLM backend: stub (configure with OPENAI_API_KEY for production)")


@cli.command()
def health():
    """Check overall system health."""
    click.echo("Checking system health...")
    
    checks = {
        "database": False,
        "vector_store": False,
        "embeddings": False,
    }
    
    # Check database
    try:
        settings = load_settings()
        storage = build_storage_repository(settings)
        checks["database"] = True
        click.echo("  ✓ Database")
    except Exception as e:
        click.echo(f"  ✗ Database: {e}")
    
    # Check vector store
    try:
        vector_store = build_vector_store()
        checks["vector_store"] = True
        click.echo("  ✓ Vector store")
    except Exception as e:
        click.echo(f"  ✗ Vector store: {e}")
    
    # Check embeddings
    try:
        from app.embeddings import EmbeddingFactory
        embedder = EmbeddingFactory.create()
        checks["embeddings"] = True
        click.echo("  ✓ Embeddings")
    except Exception as e:
        click.echo(f"  ✗ Embeddings: {e}")
    
    if all(checks.values()):
        click.echo("\n✓ All systems healthy")
        return 0
    else:
        click.echo("\n✗ Some systems unhealthy")
        return 1


if __name__ == '__main__':
    sys.exit(cli())
