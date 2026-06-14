from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

from examples.llama_50k_until_429 import (
    CANCELLED_CODE,
    MISSING_USAGE_CODE,
    MODEL,
    OLLAMA_BASE_URL,
    PROVIDER,
    QUOTA_ERROR_CODE,
    QUOTA_LIMIT,
    ProbeState,
    StreamAttempt,
    build_config,
    parse_sse_line,
    probe_passed,
)


def test_generated_config_contains_exact_llama_provider_and_quota(tmp_path: Path) -> None:
    db_path = tmp_path / "probe.sqlite3"
    config = build_config(db_path, 9876)

    assert config["server"] == {"host": "127.0.0.1", "port": 9876}
    assert config["database"] == {"url": f"sqlite:///{db_path}"}
    assert config["providers"]["default"] == PROVIDER
    assert config["providers"][PROVIDER] == {
        "type": "openai_compatible",
        "default_model": MODEL,
        "base_url": OLLAMA_BASE_URL,
    }
    assert config["models"] == {MODEL: {"output_token_limit": QUOTA_LIMIT}}

    reloaded = yaml.safe_load(yaml.safe_dump(config))
    assert reloaded == config


def test_sse_parser_detects_request_cancelled() -> None:
    attempt = StreamAttempt(phase="probe", trace_id="trace", span_name="span")

    parse_sse_line(
        attempt,
        "data: "
        + json.dumps({"error": {"message": "cancelled", "type": "cancelled", "code": CANCELLED_CODE}}),
    )

    assert attempt.terminal_code == CANCELLED_CODE
    assert attempt.cancelled is True


def test_sse_parser_treats_stream_usage_missing_as_fatal_terminal_code() -> None:
    attempt = StreamAttempt(phase="probe", trace_id="trace", span_name="span")

    parse_sse_line(
        attempt,
        "data: "
        + json.dumps(
            {
                "warning": {
                    "message": "missing usage",
                    "type": "quota_accounting",
                    "code": MISSING_USAGE_CODE,
                }
            }
        ),
    )

    assert attempt.quota_warning == MISSING_USAGE_CODE
    assert attempt.terminal_code == MISSING_USAGE_CODE


def test_sse_parser_captures_stream_usage_delta() -> None:
    attempt = StreamAttempt(phase="probe", trace_id="trace", span_name="span")

    parse_sse_line(attempt, 'data: {"choices":[],"usage":{"completion_tokens":123}}')

    assert attempt.usage_delta == 123


def test_final_result_logic_requires_circuit_open_cancel_and_final_429() -> None:
    async def build_state() -> ProbeState:
        return ProbeState(trace_id="trace", started_at=0, timeout=1)

    state = asyncio.run(build_state())
    state.observed_circuit_open = True
    state.observed_stream_cancel = True
    state.observed_final_429 = True
    assert probe_passed(state) is True

    state.observed_stream_cancel = False
    assert probe_passed(state) is False

    state.observed_stream_cancel = True
    state.observed_final_429 = False
    assert probe_passed(state) is False

    state.observed_final_429 = True
    state.failure = "stream_usage_missing"
    assert probe_passed(state) is False


def test_output_quota_code_constant_matches_expected_429_code() -> None:
    assert QUOTA_ERROR_CODE == "output_quota_exceeded"
