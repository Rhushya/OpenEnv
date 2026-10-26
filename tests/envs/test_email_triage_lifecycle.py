"""Session isolation and terminal accounting regressions."""

import random

import pytest
from envs.email_triage_env.models import EmailTriageAction
from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "adversarial"])
def test_seeded_session_does_not_change_another_sessions_next_reset(difficulty):
    reference = EmailTriageEnvironment(difficulty=difficulty)
    reference.reset(seed=77)
    expected = reference.reset().model_dump()

    session = EmailTriageEnvironment(difficulty=difficulty)
    session.reset(seed=77)
    other = EmailTriageEnvironment(difficulty=difficulty)
    other.reset(seed=123)
    other.step(EmailTriageAction(category="billing", priority=3, should_escalate=False))

    assert session.reset().model_dump() == expected


def test_reset_preserves_process_random_state():
    state = random.getstate()
    try:
        EmailTriageEnvironment().reset(seed=77)
        assert random.getstate() == state
    finally:
        random.setstate(state)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "adversarial"])
def test_terminal_episode_cannot_be_scored_again(difficulty):
    env = EmailTriageEnvironment(difficulty=difficulty)
    observation = env.reset(seed=42)
    while not observation.done:
        observation = env.step(
            EmailTriageAction(category="billing", priority=3, should_escalate=False)
        )
    terminal_state = env.state.model_dump()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(
            EmailTriageAction(category="billing", priority=3, should_escalate=False)
        )
    assert env.state.model_dump() == terminal_state
    assert env.reset(seed=42).done is False
    env.step(EmailTriageAction(category="billing", priority=3, should_escalate=False))


def test_timeout_reward_is_accounted_once():
    env = EmailTriageEnvironment(difficulty="medium")
    env.reset(seed=42)
    env._max_episode_steps = 1
    first = env.step(
        EmailTriageAction(category="billing", priority=3, should_escalate=False)
    )
    timeout = env.step(
        EmailTriageAction(category="billing", priority=3, should_escalate=False)
    )
    assert timeout.done
    assert timeout.reward == -1.0
    assert env.state.total_reward == pytest.approx(first.reward + timeout.reward)
    terminal_state = env.state.model_dump()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(
            EmailTriageAction(category="billing", priority=3, should_escalate=False)
        )
    assert env.state.model_dump() == terminal_state


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_websocket_terminal_guard_and_reset(difficulty):
    from envs.email_triage_env.client import EmailTriageEnv
    from envs.email_triage_env.models import EmailTriageObservation
    from fastapi.testclient import TestClient
    from openenv.core.env_server import create_app

    app = create_app(
        lambda: EmailTriageEnvironment(difficulty=difficulty),
        EmailTriageAction,
        EmailTriageObservation,
    )
    parser = EmailTriageEnv(base_url="http://localhost:8000")
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"seed": 42}})
        result = parser._parse_result(websocket.receive_json()["data"])
        rewards = []
        step = {
            "type": "step",
            "data": {"category": "billing", "priority": 3, "should_escalate": False},
        }
        while not result.done:
            websocket.send_json(step)
            response = websocket.receive_json()
            assert response["type"] == "observation"
            result = parser._parse_result(response["data"])
            assert result.done == result.observation.done
            rewards.append(result.reward)
        assert len(rewards) == result.observation.info.get("queue_size", 1)
        websocket.send_json(step)
        response = websocket.receive_json()
        assert response["type"] == "error"
        websocket.send_json({"type": "reset", "data": {"seed": 42}})
        result = parser._parse_result(websocket.receive_json()["data"])
        assert not result.done
        websocket.send_json(step)
        assert websocket.receive_json()["type"] == "observation"
