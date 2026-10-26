"""Email client preserves the server's reward and terminal wire fields."""

import pytest
from openenv.core.env_server.serialization import serialize_observation

from email_triage_env.client import EmailTriageEnv
from email_triage_env.models import EmailTriageObservation


@pytest.mark.parametrize("reward,done", [(None, False), (0.0, False), (0.75, True)])
def test_parse_serialized_observation(reward, done):
    observation = EmailTriageObservation(
        email_id="email-1",
        subject="Invoice",
        body_snippet="Please help",
        sender="user@example.com",
        sender_domain="example.com",
        is_internal=False,
        task_id="easy",
        reward=reward,
        done=done,
        info={"task_score": 0.75},
    )
    payload = serialize_observation(observation)
    assert "reward" not in payload["observation"]
    assert "done" not in payload["observation"]
    client = EmailTriageEnv(base_url="http://localhost:8000")
    result = client._parse_result(payload)
    assert result.reward == result.observation.reward == reward
    assert result.done == result.observation.done == done
    assert result.observation.email_id == "email-1"
    assert result.observation.info == {"task_score": 0.75}
