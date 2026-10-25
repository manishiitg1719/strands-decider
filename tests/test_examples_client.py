"""The example client's two transports, with boto3 and the network stubbed.

`examples/strands/_client.py` talks to a local `strands-decider serve` over HTTP, or to a
SageMaker endpoint through `sagemaker-runtime`. These tests pin that the same request body
goes to either, and that a missing or unreachable endpoint fails with a message that says how
to fix it. They need neither boto3 nor AWS credentials: both are replaced by stubs.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

CLIENT = Path(__file__).resolve().parent.parent / "examples" / "strands" / "_client.py"

ANSWER = {
    "model": "strands-decider-2B-qwen3.5-v1-2610",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.86}},
    "usage": {"input_tokens": 86, "output_tokens": 1},
    "latency_ms": 12.5,
}


def _load_client():
    spec = importlib.util.spec_from_file_location("decider_example_client", CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BotoCoreError(Exception):
    pass


class _ClientError(Exception):
    pass


class _Runtime:
    """Stands in for the `sagemaker-runtime` client: records calls, answers every question.

    Like the real server it returns one answer per question asked, keyed by the question's
    name, so the start-up probe (`ready`) and a real question are both answered.
    """

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        asked = json.loads(kwargs["Body"])["questions"]
        answer = next(iter(ANSWER["answers"].values()))
        reply = {**ANSWER, "answers": {name: answer for name in asked}}
        return {"Body": io.BytesIO(json.dumps(reply).encode())}


@pytest.fixture
def boto(monkeypatch):
    """Install stub boto3/botocore modules; the stub client is `boto.runtime`."""
    state = types.SimpleNamespace(runtime=_Runtime(), client_args=None, config=None)

    class Config:
        def __init__(self, **kwargs):
            state.config = kwargs

    def client(service, **kwargs):
        state.client_args = (service, kwargs)
        return state.runtime

    modules = {
        "boto3": types.SimpleNamespace(client=client),
        "botocore": types.ModuleType("botocore"),
        "botocore.config": types.SimpleNamespace(Config=Config),
        "botocore.exceptions": types.SimpleNamespace(
            BotoCoreError=_BotoCoreError, ClientError=_ClientError
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("STRANDS_DECIDER_SAGEMAKER_ENDPOINT", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    return state


def test_endpoint_argument_calls_sagemaker_with_the_systemone_body(boto):
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint", region="us-west-2", timeout=30)

    answers = decider.ask("Help!", {"is_urgent": decider.noul("Does this convey urgency?")})

    assert answers == ANSWER["answers"]
    [call] = boto.runtime.calls
    assert call["EndpointName"] == "my-endpoint"
    assert call["ContentType"] == "application/json"
    body = json.loads(call["Body"])
    assert body["state"] == "Help!"
    assert body["questions"]["is_urgent"] == {
        "type": "noul",
        "instructions": "Does this convey urgency?",
    }
    service, kwargs = boto.client_args
    assert service == "sagemaker-runtime"
    assert kwargs["region_name"] == "us-west-2"
    # The client's read timeout is the caller's, and a slow request is not sent twice.
    # `total_max_attempts` counts the first call, so 1 is "no retry". `max_attempts: 1` would
    # mean one *retry*, i.e. two attempts, and send a slow request twice.
    assert boto.config == {"read_timeout": 30, "retries": {"total_max_attempts": 1}}
    assert (decider.calls, decider.input_tokens, decider.last_latency_ms) == (1, 86, 12.5)


def test_endpoint_environment_variable_selects_sagemaker(boto, monkeypatch):
    monkeypatch.setenv("STRANDS_DECIDER_SAGEMAKER_ENDPOINT", "from-env")
    client = _load_client()

    client.Decider().ask("x", {"q": client.Decider.noul("?")})

    assert boto.runtime.calls[0]["EndpointName"] == "from-env"


def test_argument_wins_over_environment(boto, monkeypatch):
    monkeypatch.setenv("STRANDS_DECIDER_SAGEMAKER_ENDPOINT", "from-env")
    client = _load_client()

    client.Decider(sagemaker_endpoint="from-arg").ask("x", {"q": client.Decider.noul("?")})

    assert boto.runtime.calls[0]["EndpointName"] == "from-arg"


def test_without_an_endpoint_the_http_path_is_unchanged(boto, monkeypatch):
    client = _load_client()
    seen = {}

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return io.BytesIO(json.dumps(ANSWER).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)

    # io.BytesIO is its own context manager, as the response `with` block expects.
    answers = client.Decider(url="http://localhost:9/").ask("x", {"q": client.Decider.noul("?")})

    assert answers == ANSWER["answers"]
    assert seen["url"] == "http://localhost:9/v1/systemone"
    assert seen["body"]["questions"]["q"]["type"] == "noul"
    assert boto.runtime.calls == []  # SageMaker was never touched
    assert boto.client_args is None  # and no boto3 client was even created


def test_sagemaker_failure_says_how_to_fix_it(boto):
    boto.runtime.fail = _ClientError("ValidationError: Endpoint my-endpoint not found")
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint", region="eu-west-1")

    with pytest.raises(client.ServerUnavailable) as raised:
        decider.ask("x", {"q": client.Decider.noul("?")})

    message = str(raised.value)
    assert "'my-endpoint' in eu-west-1" in message
    assert "deploy/sagemaker/README.md" in message
    assert "sagemaker:InvokeEndpoint" in message
    assert decider.calls == 0  # a failed request is not counted


def test_missing_boto3_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("STRANDS_DECIDER_SAGEMAKER_ENDPOINT", raising=False)
    monkeypatch.setitem(sys.modules, "boto3", None)  # makes `import boto3` raise ImportError
    client = _load_client()

    with pytest.raises(client.ServerUnavailable, match="pip install boto3"):
        client.Decider(sagemaker_endpoint="e").ask("x", {"q": client.Decider.noul("?")})


def test_sagemaker_banner_probes_once_without_counting_it(boto):
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint", region="us-west-2")

    banner = decider.banner()

    assert banner == (
        "strands-decider: strands-decider-2B-qwen3.5-v1-2610 on SageMaker endpoint "
        "my-endpoint in us-west-2"
    )
    assert len(boto.runtime.calls) == 1
    assert decider.calls == 0


def test_sagemaker_body_that_is_not_an_object_is_a_clear_error(boto):
    boto.runtime.invoke_endpoint = lambda **kwargs: {"Body": io.BytesIO(b'["not", "an", "object"]')}
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint")

    with pytest.raises(client.ServerUnavailable, match="'my-endpoint' returned a JSON list"):
        decider.ask("x", {"q": client.Decider.noul("?")})

    assert decider.calls == 0


def test_http_body_that_is_not_an_object_is_a_clear_error(boto, monkeypatch):
    client = _load_client()
    monkeypatch.setattr(
        client.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b'"oops"')
    )
    decider = client.Decider(url="http://localhost:9")

    with pytest.raises(client.ServerUnavailable, match="localhost:9 returned a JSON str"):
        decider.ask("x", {"q": client.Decider.noul("?")})


def test_answers_that_are_not_an_object_are_not_counted(boto):
    boto.runtime.invoke_endpoint = lambda **kwargs: {
        "Body": io.BytesIO(json.dumps({**ANSWER, "answers": [1, 2]}).encode())
    }
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint")

    with pytest.raises(client.ServerUnavailable, match="`answers` is a list"):
        decider.ask("x", {"q": client.Decider.noul("?")})

    assert decider.calls == 0


def _region_passed_to_boto(boto, client, **kwargs):
    client.Decider(sagemaker_endpoint="e", **kwargs).ask("x", {"q": client.Decider.noul("?")})
    return boto.client_args[1]["region_name"]


def test_region_argument_wins_over_the_environment(boto, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    client = _load_client()

    assert _region_passed_to_boto(boto, client, region="us-west-2") == "us-west-2"


def test_aws_region_wins_over_aws_default_region(boto, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    client = _load_client()

    assert _region_passed_to_boto(boto, client) == "eu-west-1"


def test_aws_default_region_is_used_when_aws_region_is_unset(boto, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    client = _load_client()

    assert _region_passed_to_boto(boto, client) == "ap-south-1"


def test_no_region_anywhere_leaves_it_to_boto3_and_the_profile(boto):
    client = _load_client()

    # None, not "": boto3 then resolves the region of the active AWS profile itself.
    assert _region_passed_to_boto(boto, client) is None


def test_missing_region_at_client_construction_is_a_clear_error(boto, monkeypatch):
    def client_without_region(service, **kwargs):
        raise _BotoCoreError("You must specify a region.")

    monkeypatch.setattr(sys.modules["boto3"], "client", client_without_region, raising=False)
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint")

    with pytest.raises(client.ServerUnavailable) as raised:
        decider.ask("x", {"q": client.Decider.noul("?")})

    message = str(raised.value)
    assert "'my-endpoint'" in message
    assert "You must specify a region." in message
    assert "AWS_DEFAULT_REGION" in message
    assert decider.calls == 0
    assert decider._runtime is None  # nothing half-built is cached for the next call


def _probe_decider(boto, client, body: bytes):
    boto.runtime.invoke_endpoint = lambda **kwargs: {"Body": io.BytesIO(body)}
    return client.Decider(sagemaker_endpoint="my-endpoint", region="us-west-2")


def _probe(**overrides):
    reply = {**ANSWER, "answers": {"ready": {"type": "noul", "noul": 0.1}}, **overrides}
    return json.dumps({k: v for k, v in reply.items() if v is not None}).encode()


def test_health_probe_accepts_a_real_reply_and_is_not_counted(boto):
    client = _load_client()
    decider = _probe_decider(boto, client, _probe())

    assert decider.health() == {
        "checkpoint": "strands-decider-2B-qwen3.5-v1-2610",
        "device": "sagemaker",
    }
    assert decider.calls == 0 and decider.input_tokens == 0


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (b'{"model": "m"}', "without an `answers` object (got NoneType)"),
        (b'{"model": "m", "answers": ["ready"]}', "(got list)"),
        (b'{"model": "m", "answers": "ready"}', "(got str)"),
        (b'{"model": "m", "answers": {}}', "not with the expected noul answer"),
        (b'{"model": "m", "answers": {"ready": "yes"}}', "not with the expected noul answer"),
        (
            b'{"model": "m", "answers": {"ready": {"type": "noul", "noul": "high"}}}',
            "not with the expected noul answer",
        ),
        (
            b'{"model": "m", "answers": {"ready": {"type": "noul", "noul": true}}}',
            "not with the expected noul answer",
        ),
        (_probe(model=""), "without a `model` name"),
        (b'{"answers": {"ready": {"type": "noul", "noul": 0.1}}}', "without a `model` name"),
    ],
)
def test_health_probe_rejects_a_reply_that_is_not_a_decider_answer(boto, body, fragment):
    client = _load_client()
    decider = _probe_decider(boto, client, body)

    with pytest.raises(client.ServerUnavailable, match="'my-endpoint'") as raised:
        decider.health()

    assert fragment in str(raised.value)
    assert decider.calls == 0


@pytest.mark.parametrize("noul", ["-0.1", "1.1", "2", "-1", "NaN", "Infinity", "-Infinity"])
def test_health_probe_rejects_a_noul_that_is_not_a_probability(boto, noul):
    client = _load_client()
    body = (
        b'{"model": "m", "usage": {"input_tokens": 3}, '
        b'"answers": {"ready": {"type": "noul", "noul": %s}}}' % noul.encode()
    )
    decider = _probe_decider(boto, client, body)

    with pytest.raises(client.ServerUnavailable, match="a finite number from 0 to 1"):
        decider.health()

    assert (decider.calls, decider.input_tokens) == (0, 0)


def test_health_probe_rejects_a_huge_integer_without_overflowing(boto):
    # `math.isfinite(10**400)` raises OverflowError; the range check must come first.
    client = _load_client()
    huge = b"1" + b"0" * 399  # a 400-digit integer
    body = (
        b'{"model": "m", "usage": {"input_tokens": 3}, '
        b'"answers": {"ready": {"type": "noul", "noul": %s}}}' % huge
    )
    decider = _probe_decider(boto, client, body)

    with pytest.raises(client.ServerUnavailable, match="a finite number from 0 to 1"):
        decider.health()

    assert (decider.calls, decider.input_tokens) == (0, 0)


@pytest.mark.parametrize("noul", ["0", "1", "0.0", "1.0", "0.5"])
def test_health_probe_accepts_the_whole_inclusive_range(boto, noul):
    client = _load_client()
    body = b'{"model": "m", "answers": {"ready": {"type": "noul", "noul": %s}}}' % noul.encode()
    decider = _probe_decider(boto, client, body)

    assert decider.health() == {"checkpoint": "m", "device": "sagemaker"}
    assert (decider.calls, decider.input_tokens) == (0, 0)


@pytest.mark.parametrize("body", [b'["not", "an", "object"]', b'"oops"'])
def test_health_probe_rejects_a_json_body_that_is_not_an_object(boto, body):
    client = _load_client()
    decider = _probe_decider(boto, client, body)

    with pytest.raises(client.ServerUnavailable, match="returned a JSON"):
        decider.health()


@pytest.mark.parametrize("body", [b"<html>502 Bad Gateway</html>", b"", b"\xff\xfe"])
def test_health_probe_rejects_a_body_that_is_not_json(boto, body):
    client = _load_client()
    decider = _probe_decider(boto, client, body)

    with pytest.raises(client.ServerUnavailable, match="body that is not JSON"):
        decider.health()


def test_banner_does_not_report_success_for_a_bad_probe(boto):
    client = _load_client()
    decider = _probe_decider(boto, client, b'{"model": "m"}')

    with pytest.raises(client.ServerUnavailable):
        decider.banner()


def test_banner_reports_the_region_boto3_resolved_from_the_profile(boto):
    boto.runtime.meta = types.SimpleNamespace(region_name="ap-southeast-2")  # from the profile
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint")  # no region=, no AWS_* region

    assert decider.region is None
    banner = decider.banner()

    assert boto.client_args[1]["region_name"] is None  # boto3 was left to resolve it
    assert banner.endswith("on SageMaker endpoint my-endpoint in ap-southeast-2")
    assert decider.region == "ap-southeast-2"


def test_explicit_region_is_not_replaced_by_the_client_s_region(boto):
    boto.runtime.meta = types.SimpleNamespace(region_name="ap-southeast-2")
    client = _load_client()
    decider = client.Decider(sagemaker_endpoint="my-endpoint", region="us-west-2")

    assert decider.banner().endswith("my-endpoint in us-west-2")


def test_banner_has_no_region_when_the_client_reports_none(boto):
    client = _load_client()  # the stub runtime has no `meta` at all
    decider = client.Decider(sagemaker_endpoint="my-endpoint")

    assert decider.banner().endswith("on SageMaker endpoint my-endpoint")
