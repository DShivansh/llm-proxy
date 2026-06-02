from pathlib import Path

from fastapi.testclient import TestClient

from llm_proxy.app import create_app
from llm_proxy.config import load_config


def write_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "llm_proxy.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
server:
  host: 127.0.0.1
  port: 8000
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


def chat(client: TestClient, span_name: str, trace_id: str = "launch-demo-001"):
    return client.post(
        "/v1/chat/completions",
        headers={
            "X-Trace-ID": trace_id,
            "X-Span-Name": span_name,
            "X-LLM-Provider": "mock",
        },
        json={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Plan the llm proxy launch."}],
        },
    )


def test_mock_workflow_opens_circuit_and_records_trace(tmp_path: Path):
    app = create_app(load_config(write_config(tmp_path)))

    with TestClient(app) as client:
        expected_outputs = {
            "planner_agent": 120,
            "researcher_agent": 130,
            "writer_agent": 160,
            "reviewer_agent": 120,
        }

        for span_name, output_tokens in expected_outputs.items():
            response = chat(client, span_name)
            assert response.status_code == 200
            body = response.json()
            assert body["object"] == "chat.completion"
            assert body["usage"]["completion_tokens"] == output_tokens

        blocked = chat(client, "final_editor_agent")
        assert blocked.status_code == 429
        assert blocked.json() == {
            "error": {
                "message": "Output token circuit is open for model gpt-4.1-mini",
                "type": "quota_exceeded",
                "code": "output_quota_exceeded",
            }
        }

        usage = client.get("/internal/usage/models").json()
        assert usage == [
            {
                "model": "gpt-4.1-mini",
                "period_key": "manual",
                "output_token_limit": 500,
                "output_tokens_used": 530,
                "circuit_open": True,
            }
        ]

        trace = client.get("/internal/traces/launch-demo-001").json()
        assert trace["trace_id"] == "launch-demo-001"
        assert [span["span_name"] for span in trace["spans"]] == [
            "planner_agent",
            "researcher_agent",
            "writer_agent",
            "reviewer_agent",
            "final_editor_agent",
        ]
        assert [span["status"] for span in trace["spans"]] == [
            "success",
            "success",
            "success",
            "success",
            "blocked",
        ]
        assert trace["spans"][3]["circuit_opened_by_this_span"] is True
        assert trace["spans"][4]["http_status_code"] == 429

        summaries = client.get("/internal/traces").json()
        assert summaries[0].pop("created_at").endswith("Z")
        assert summaries[0].pop("updated_at").endswith("Z")
        assert summaries == [
            {
                "trace_id": "launch-demo-001",
                "span_count": 5,
                "status": "blocked",
                "models": ["gpt-4.1-mini"],
                "total_input_tokens": 140,
                "total_output_tokens": 530,
            }
        ]


def test_streaming_requests_are_rejected_and_traced(tmp_path: Path):
    app = create_app(load_config(write_config(tmp_path)))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Trace-ID": "stream-test", "X-Span-Name": "streaming_call"},
            json={
                "model": "gpt-4.1-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "Stream this."}],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "streaming_not_supported"

        trace = client.get("/internal/traces/stream-test").json()
        assert trace["spans"][0]["status"] == "error"
        assert trace["spans"][0]["error_message"] == "Streaming is not supported by this proxy MVP"
