"""The Alembic migrations must produce exactly the schema the ORM models describe."""

import os
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, inspect

from vlm_inspect.api.db import Base

ROOT = Path(__file__).resolve().parents[1]


def _database_url(tmp_path: Path) -> str:
    # CI sets VLM_INSPECT_TEST_DATABASE_URL to a PostgreSQL service; locally SQLite is used.
    return (
        os.environ.get("VLM_INSPECT_TEST_DATABASE_URL")
        or f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    )


def test_upgrade_matches_models_and_downgrade_is_clean(tmp_path, monkeypatch) -> None:
    url = _database_url(tmp_path)
    monkeypatch.setenv("VLM_INSPECT_DATABASE_URL", url)
    config = Config(str(ROOT / "alembic.ini"))

    command.upgrade(config, "head")
    engine = create_engine(url)

    def compare_type(context, inspected, metadata, inspected_type, metadata_type):
        # SQLite has no vector type and reflects VECTOR(384) as NUMERIC(384); pgvector columns are
        # only meaningful (and fully compared) on PostgreSQL.
        if engine.dialect.name == "sqlite" and isinstance(metadata_type, Vector):
            return False
        return None  # default comparison

    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": compare_type})
        diff = compare_metadata(context, Base.metadata)
    assert diff == [], f"migrations and models disagree: {diff}"

    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    engine.dispose()
