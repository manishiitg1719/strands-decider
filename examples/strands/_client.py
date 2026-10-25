"""A small synchronous client for a running decision-model server.

Deliberately `urllib` rather than a client library: these examples should need nothing
beyond `strands-agents`, so that a reader can drop one into their own project without
first agreeing to a dependency.

Synchronous on purpose too. Strands calls a handler method that is a plain `def`
directly, so a blocking request is the simplest thing that works -- and it keeps requests
to the server strictly sequential, which is how it is best driven: one request at a time.

Two transports, one wire body. By default the client posts to `/v1/systemone` on a server you
started with `strands-decider serve`. Set `STRANDS_DECIDER_SAGEMAKER_ENDPOINT` (or pass
`sagemaker_endpoint=`) and it calls that SageMaker real-time endpoint instead, through
`sagemaker-runtime`'s `InvokeEndpoint`, signed with your AWS credentials
(`deploy/sagemaker/README.md`). Only the transport changes: the questions, the answers and
the rest of the example are the same either way.

**Renaming.** The product name is spelled out in exactly one place, `NAME` below, and the
things derived from it sit directly underneath. Everything else -- this module's filename,
the `Decider` class, the demos -- is named for what it does rather than what it is called,
so a future rename is an edit to this block and nothing else.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

# --------------------------------------------------------------------------- the name

NAME = "strands-decider"
"""What this thing is called today. The only place the name is written."""

ENV_URL = "STRANDS_DECIDER_URL"
"""Environment variable that overrides the endpoint. Derived from NAME, so rename together."""

ENV_SAGEMAKER_ENDPOINT = "STRANDS_DECIDER_SAGEMAKER_ENDPOINT"
"""Environment variable naming a SageMaker endpoint to call instead of `ENV_URL`'s server."""

ENV_REGIONS = ("AWS_REGION", "AWS_DEFAULT_REGION")
"""Environment variables read for the AWS region, in order, when `region=` is not given.

boto3 itself only documents `AWS_DEFAULT_REGION`; `AWS_REGION` is what most other AWS tooling
and runtimes (Lambda, the AWS CLI's docs, the other SDKs) set, so both are honoured here.
"""

SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
"""How to start a server, shown when nothing is listening.

This is the upstream CLI entry point (`[project.scripts]` in `pyproject.toml`), which is a
separate thing from NAME: it only changes if the Python package itself is renamed.
"""

DEPLOY_HINT = "see deploy/sagemaker/README.md (`npx cdk deploy`)"
"""How to create the SageMaker endpoint, shown when it cannot be called."""

MODEL_ID = "hobson-latest"
"""Sent as `model` on every request. The server ignores it and answers as whatever
checkpoint it loaded, so this is only here to show the full wire shape."""

DEFAULT_PORT = 8099
DEFAULT_URL = os.environ.get(ENV_URL, f"http://127.0.0.1:{DEFAULT_PORT}")


class ServerUnavailable(RuntimeError):
    """The server is not answering. Raised with the command that starts one."""


def _json_object(raw: bytes | str, source: str) -> dict[str, Any]:
    """Decode a response body, which the server contract says is a JSON object.

    `json.loads` is typed as returning `Any`, so the shape is checked here rather than assumed:
    a body that is valid JSON but not an object (a bare string or list, say, from a proxy or a
    wrong endpoint) fails with a message that names where it came from. A body that is not
    JSON at all still raises `json.JSONDecodeError`, as before.
    """
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ServerUnavailable(
            f"{source} returned a JSON {type(decoded).__name__}, expected an object"
        )
    return decoded


def _probe_checkpoint(payload: dict[str, Any], source: str) -> str:
    """The checkpoint name from a health-probe response, which must have the full reply shape.

    The server always answers with a `model` string and an `answers` object holding one answer
    per question (`SystemOneResponse` in `strands_decider/schema.py`). A reply without them is
    not a decider answering, so it is an error and not a banner with a made-up checkpoint.
    """
    model = payload.get("model")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ServerUnavailable(
            f"{source} answered the start-up probe without an `answers` object "
            f"(got {type(answers).__name__}); is it a {NAME} endpoint?"
        )
    ready = answers.get("ready")
    value = ready.get("noul") if isinstance(ready, dict) else None
    if (
        not isinstance(ready, dict)
        or ready.get("type") != "noul"
        or isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1  # first: comparing a huge int is safe, `math.isfinite` is not
        or not math.isfinite(value)
    ):
        raise ServerUnavailable(
            f"{source} answered the start-up probe, but not with the expected noul answer "
            f"for 'ready' (a probability: a finite number from 0 to 1); is it a {NAME} endpoint?"
        )
    if not isinstance(model, str) or not model:
        raise ServerUnavailable(
            f"{source} answered the start-up probe without a `model` name; is it a {NAME} endpoint?"
        )
    return model


class Decider:
    """One state, many typed questions, one round trip.

    The questions cannot see each other, so their answers stay independent -- but the
    state is encoded once and shared across them, which is why asking eight things costs
    far less than eight requests. Ask everything you need at once.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout: float = 120.0,
        *,
        sagemaker_endpoint: str | None = None,
        region: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        # An endpoint name, from the argument or the environment, selects SageMaker. The region
        # is the `region=` argument, else AWS_REGION, else AWS_DEFAULT_REGION. If none is set it
        # stays None and boto3 falls back to the region of the active AWS profile.
        self.sagemaker_endpoint = sagemaker_endpoint or os.environ.get(ENV_SAGEMAKER_ENDPOINT)
        self.region = region or next(
            (value for name in ENV_REGIONS if (value := os.environ.get(name))), None
        )
        self._runtime: Any = None
        # Generous: on Apple silicon the *first* request at a given input length pays a
        # one-off MPS shape compile, which is seconds on a long document.
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None

    # ---- questions -------------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true), and with two outcomes that *is* the confidence.

        `criteria` optionally describes what `"true"` and `"false"` mean. It is worth
        writing: a noul renders as a two-option list, so those descriptions are read the
        same way a choice's options are, and they sharpen the boundary considerably.
        """
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means.

        The option names are read from this request, not baked into the model, so you can
        change them without retraining anything.
        """
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ----------------------------------------------------------

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        if self.sagemaker_endpoint:
            where = f"SageMaker endpoint {self.sagemaker_endpoint!r}"
            if self.region:
                where += f" in {self.region}"
            return ServerUnavailable(
                f"cannot call {where} ({exc}). It needs a deployed endpoint "
                f"({DEPLOY_HINT}), a region (region=, AWS_REGION, AWS_DEFAULT_REGION or "
                "your AWS profile's region) and credentials with sagemaker:InvokeEndpoint on it."
            )
        return ServerUnavailable(
            f"no {NAME} at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def _sagemaker_runtime(self) -> Any:
        """The `sagemaker-runtime` client, created on first use.

        boto3 is imported here and not at the top so the plain-HTTP path still needs nothing
        beyond `strands-agents` (which depends on boto3 anyway, for Bedrock).
        """
        if self._runtime is None:
            try:
                import boto3
                from botocore.config import Config
                from botocore.exceptions import BotoCoreError
            except ImportError as exc:
                raise ServerUnavailable(
                    f"calling a SageMaker endpoint needs boto3 ({exc}): pip install boto3"
                ) from exc
            # `read_timeout` is how long the client waits for response data on the connection
            # (botocore's default is 60 s). A cold endpoint or a long document can take longer
            # than that to answer, so it follows the caller's timeout. It is not an end-to-end
            # deadline on the request, and `connect_timeout` is a separate limit that is left
            # at botocore's default (60 s). `total_max_attempts` counts the
            # first call, so 1 means no retry and a slow request is never silently sent twice
            # (`max_attempts` counts *retries*: `{"max_attempts": 1}` would allow two attempts).
            config = Config(read_timeout=self.timeout, retries={"total_max_attempts": 1})
            try:
                # Building the client resolves the region and the profile, so a missing region
                # (NoRegionError) or an unknown profile fails here, not as a stray exception.
                self._runtime = boto3.client(
                    "sagemaker-runtime", region_name=self.region, config=config
                )
            except BotoCoreError as exc:
                raise self._unavailable(exc) from exc
            # With no region from the argument or the environment, boto3 resolved one from the
            # active AWS profile. Keep it, so errors and `banner()` show the region really used.
            effective = getattr(getattr(self._runtime, "meta", None), "region_name", None)
            if not self.region and isinstance(effective, str) and effective:
                self.region = effective
        return self._runtime

    def _post(self, body: bytes) -> dict[str, Any]:
        """Send one `/v1/systemone` body over whichever transport is configured."""
        if self.sagemaker_endpoint:
            runtime = self._sagemaker_runtime()  # raises a clear error if boto3 is missing
            from botocore.exceptions import BotoCoreError, ClientError

            try:
                response = runtime.invoke_endpoint(
                    EndpointName=self.sagemaker_endpoint,
                    ContentType="application/json",
                    Body=body,
                )
                return _json_object(
                    response["Body"].read(), f"SageMaker endpoint {self.sagemaker_endpoint!r}"
                )
            except (BotoCoreError, ClientError) as exc:
                raise self._unavailable(exc) from exc
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return _json_object(response.read(), self.url)
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers come back under the same keys."""
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        payload = self._post(body)
        answers = payload["answers"]
        if not isinstance(answers, dict):
            raise ServerUnavailable(
                f"the server's `answers` is a {type(answers).__name__}, expected an object"
            )
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return answers

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving.

        A SageMaker endpoint exposes only `/invocations`, so there is no `/health` to read.
        The probe is one real one-question request (it is not counted in `calls`), which
        checks the endpoint, the region and the credentials the same way the first real
        question will, and needs no permission beyond `sagemaker:InvokeEndpoint`. The reply
        is checked too: it must carry a `model` and a noul answer for the probe question.
        """
        if self.sagemaker_endpoint:
            probe = {"ready": self.noul("Is this a test message?")}
            body = json.dumps({"state": "ping", "model": MODEL_ID, "questions": probe}).encode()
            where = f"SageMaker endpoint {self.sagemaker_endpoint!r}"
            try:
                payload = self._post(body)
            except ValueError as exc:  # not JSON at all (json.JSONDecodeError, bad UTF-8)
                raise ServerUnavailable(
                    f"{where} answered the start-up probe with a body that is not JSON ({exc})"
                ) from exc
            return {"checkpoint": _probe_checkpoint(payload, where), "device": "sagemaker"}
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return _json_object(response.read(), f"{self.url}/health")
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        h = self.health()
        if self.sagemaker_endpoint:
            region = f" in {self.region}" if self.region else ""
            return (
                f"{NAME}: {h['checkpoint']} on SageMaker endpoint {self.sagemaker_endpoint}{region}"
            )
        return (
            f"{NAME}: {h['checkpoint']} on {h['base_model']} "
            f"[{h['device']}, window {h['max_length']}] at {self.url}"
        )

    def __str__(self) -> str:
        return f"{self.calls} decisions, {self.input_tokens} input tokens"
