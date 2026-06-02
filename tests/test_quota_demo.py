from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import uvicorn

from llm_proxy.app import create_app
from llm_proxy.config import load_config


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "demo.db"
    config_path.write_text(
        f"""
server:
  host: 127.0.0.1
  port: 8765
database:
  url: sqlite:///{db_path}
tracing:
  log_full_body: true
quota:
  period: manual
providers:
  default: mock
  mock:
    type: smart_mock
    default_model: gpt-4.1-mini
models:
  gpt-4.1-mini:
    output_token_limit: 500
""",
        encoding="utf-8",
    )
    return config_path


def test_quota_demo_prints_workflow_summary(tmp_path: Path):
    config = uvicorn.Config(
        create_app(load_config(write_config(tmp_path))),
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass

    try:
        result = subprocess.run(
            [
                sys.executable,
                "examples/quota_demo.py",
                "--base-url",
                "http://127.0.0.1:8765",
                "--trace-id",
                "launch-demo-001",
            ],
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert "Trace: launch-demo-001" in result.stdout
    assert "planner_agent       success   output=120   used=120/500" in result.stdout
    assert "reviewer_agent      success   output=120   used=530/500 circuit opened" in result.stdout
    assert "final_editor_agent  blocked   circuit already open" in result.stdout
    assert "http://127.0.0.1:8765/internal/traces/launch-demo-001" in result.stdout
