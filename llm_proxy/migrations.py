from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def alembic_config(database_url: str) -> Config:
    config = Config(project_root() / "alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def run_migrations(database_url: str) -> None:
    command.upgrade(alembic_config(database_url), "head")
