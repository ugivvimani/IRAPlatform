"""
CLI management commands for IRA platform.
Run with: python -m app.cli <command> <args>
"""
import sys
import logging
from datetime import datetime, timezone

import click

from app.core.config import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store

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
    build_storage_repository(settings)
    click.echo(f"Database initialized (backend: {settings.db_backend})")


@db.command()
def migrate():
    """Run database migrations."""
    click.echo("Running migrations...")
    click.echo("Use 'alembic upgrade head' to run Alembic migrations.")


@cli.group()
def vector():
    """Vector store commands."""
    pass


@vector.command("health")
def vector_health():
    """Check vector store health."""
    click.echo("Checking vector store health...")
    try:
        vs = build_vector_store()
        click.echo(f"Vector store healthy: {type(vs).__name__}")
    except Exception as e:
        click.echo(f"Vector store error: {e}")
        sys.exit(1)


@vector.command()
@click.option("--query", prompt="Query text", help="Text to embed")
def embed(query):
    """Test embedding functionality."""
    click.echo(f"Embedding query: '{query}'...")
    try:
        from app.services.embeddings import EmbeddingFactory
        embedder = EmbeddingFactory.create()
        embedding = embedder.embed_sync([query])
        click.echo(f"Dimensions: {len(embedding[0])}")
        click.echo(f"First 5 values: {embedding[0][:5]}")
    except Exception as e:
        click.echo(f"Embedding error: {e}")
        sys.exit(1)


@cli.group()
def watchlist():
    """Watchlist commands."""
    pass


@watchlist.command("list")
def watchlist_list():
    """List all watchlisted entities."""
    click.echo("Fetching watchlist...")
    settings = load_settings()
    storage = build_storage_repository(settings)
    entries = storage.list_watchlist()
    if not entries:
        click.echo("Watchlist is empty.")
        return
    click.echo(f"Found {len(entries)} entities on watchlist:")
    for entry in entries:
        click.echo(f"  - {entry.entity_id}: {entry.company_name}")


@watchlist.command()
@click.option("--entity-id", prompt="Entity ID", help="Entity ID")
@click.option("--company-name", prompt="Company name", help="Company name")
@click.option("--notes", default="", help="Notes")
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
    click.echo(f"Added: {result.entity_id} / {result.company_name}")


@watchlist.command()
@click.option("--entity-id", prompt="Entity ID", help="Entity ID")
def remove(entity_id):
    """Remove entity from watchlist."""
    click.echo(f"Removing '{entity_id}' from watchlist...")
    settings = load_settings()
    storage = build_storage_repository(settings)
    success = storage.delete_watchlist(entity_id)
    if success:
        click.echo("Entity removed.")
    else:
        click.echo("Entity not found.")
        sys.exit(1)


@cli.group()
def assessment():
    """Assessment commands."""
    pass


@assessment.command()
@click.option("--entity-id", prompt="Entity ID", help="Entity ID")
@click.option("--limit", default=10, help="Number of assessments to show")
def history(entity_id, limit):
    """Show assessment history for an entity."""
    click.echo(f"Fetching assessment history for '{entity_id}'...")
    settings = load_settings()
    storage = build_storage_repository(settings)
    records = storage.list_assessments(entity_id=entity_id, limit=limit)
    if not records:
        click.echo("No assessments found.")
        return
    click.echo(f"Found {len(records)} assessments:")
    for record in records:
        click.echo(f"  - {record.created_at}: {record.risk_rating} (confidence: {record.confidence})")


@cli.group("config")
def config_group():
    """Configuration commands."""
    pass


@config_group.command("show")
def config_show():
    """Show current configuration."""
    settings = load_settings()
    click.echo("Current configuration:")
    click.echo(f"  App environment: {settings.app_env}")
    click.echo(f"  Database backend: {settings.db_backend}")
    click.echo(f"  Vector backend: {settings.vector_backend}")


@cli.command("health")
def system_health():
    """Check overall system health."""
    click.echo("Checking system health...")
    checks: dict[str, bool] = {"database": False, "vector_store": False, "embeddings": False}

    try:
        settings = load_settings()
        build_storage_repository(settings)
        checks["database"] = True
        click.echo("  Database: OK")
    except Exception as e:
        click.echo(f"  Database: ERROR - {e}")

    try:
        build_vector_store()
        checks["vector_store"] = True
        click.echo("  Vector store: OK")
    except Exception as e:
        click.echo(f"  Vector store: ERROR - {e}")

    try:
        from app.services.embeddings import EmbeddingFactory
        EmbeddingFactory.create()
        checks["embeddings"] = True
        click.echo("  Embeddings: OK")
    except Exception as e:
        click.echo(f"  Embeddings: ERROR - {e}")

    if all(checks.values()):
        click.echo("\nAll systems healthy.")
        sys.exit(0)
    else:
        click.echo("\nSome systems unhealthy.")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(cli())
