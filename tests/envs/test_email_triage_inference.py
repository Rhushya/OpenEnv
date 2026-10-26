"""Offline inference regressions with deterministic responses and a local server."""

import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import uvicorn
from envs.email_triage_env import inference
from envs.email_triage_env.models import EmailTriageAction, EmailTriageObservation
from envs.email_triage_env.server import email_triage_environment as runtime
from openenv.core.env_server import create_app


@pytest.mark.parametrize(
    "value, expected",
    [
        (False, False),
        (True, True),
        ("false", False),
        ("true", True),
        (" FALSE ", False),
    ],
)
def test_parse_escalation(value, expected):
    action = inference._parse_model_action(
        json.dumps({"category": "spam", "priority": 1, "should_escalate": value}),
        "Newsletter",
        "Weekly news",
    )
    assert action == inference.ParsedAction("spam", 1, expected)


@pytest.mark.parametrize("value", ["maybe", 1, [], None])
def test_invalid_escalation_uses_existing_heuristic_fallback(value):
    action = inference._parse_model_action(
        json.dumps({"category": "spam", "priority": 1, "should_escalate": value}),
        "Production outage",
        "Critical incident",
    )
    assert action == inference.ParsedAction("urgent", 5, True)


@pytest.fixture
def local_server():
    env = runtime.EmailTriageEnvironment()
    app = create_app(lambda: env, EmailTriageAction, EmailTriageObservation)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield base_url, env
        finally:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "adversarial"])
def test_inference_completes_queue_and_scores_every_ticket(
    local_server, difficulty, capsys
):
    base_url, env = local_server
    prompts = []

    def complete(**kwargs):
        observation = env._current_email
        prompt = kwargs["messages"][1]["content"]
        assert observation["subject"] in prompt
        prompts.append(prompt)
        # Deterministic fake model: exercise JSON parsing without a provider call.
        content = json.dumps(
            {
                "category": observation["true_category"],
                "priority": observation["true_priority"],
                "should_escalate": str(observation["needs_escalation"]).lower(),
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
    )
    score = inference._run_task(base_url, client, difficulty, seed=42)
    assert env._done
    assert env.state.step_count == len(env._queue) == len(prompts)
    assert score == pytest.approx(env.state.total_reward)
    output = capsys.readouterr().out
    assert output.count("[STEP]") == len(prompts)
    assert f"steps={len(prompts)} score={score:.3f}" in output
    assert "done=true" in output


def test_step_failure_does_not_report_partial_episode_as_success(
    local_server, monkeypatch, capsys
):
    base_url, _ = local_server
    post = requests.post
    steps = 0

    def fail_second_step(url, **kwargs):
        nonlocal steps
        if url.endswith("/step"):
            steps += 1
            if steps == 2:
                raise requests.ConnectionError("connection lost")
        return post(url, **kwargs)

    monkeypatch.setattr(inference.requests, "post", fail_second_step)
    score = inference._run_task(base_url, None, "medium", 42)
    assert score > 0.0
    assert steps == 2
    assert "[END] success=false" in capsys.readouterr().out


def test_easy_task_grader_runs_once_without_changing_reward(monkeypatch):
    grader = Mock(wraps=runtime.task_grader)
    monkeypatch.setattr(runtime, "task_grader", grader)
    env = runtime.EmailTriageEnvironment(difficulty="easy")
    env.reset(seed=42)
    email = env._current_email
    result = env.step(
        EmailTriageAction(
            category=email["true_category"],
            priority=email["true_priority"],
            should_escalate=email["needs_escalation"],
        )
    )
    assert grader.call_count == 1
    assert result.info["task_score"] == 1.0
    assert result.reward == pytest.approx(1.35)
