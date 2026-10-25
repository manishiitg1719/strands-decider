# Strands decider inside a Strands agent

A worked example of strands decider inside a [Strands](https://github.com/strands-agents/sdk-python)
agent. The agent runs locally on the default Bedrock model plus the locally served decider. It
runs from a clone of the repository and expects a server on port 8099 unless
`STRANDS_DECIDER_URL` is set.

**Prerequisites:** Strands drives the agent with an LLM call to Amazon Bedrock, so you need
AWS credentials with Bedrock access in your environment before you run the example. See the
[Strands Bedrock model provider docs](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/amazon-bedrock/)
for the supported credential options.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e . strands-agents
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

python examples/strands/tool_call_intervention.py
```

On an Apple-silicon Mac, `pip install -e ".[mlx]" strands-agents` and `--device mlx` on the
`serve` line run the decider through MLX
([Serving on a Mac with MLX](../../docs/inference.md#serving-on-a-mac-with-mlx)).

## Against a hosted endpoint

The same example runs against a decider you have deployed on Amazon SageMaker
([deploy/sagemaker/README.md](../../deploy/sagemaker/README.md)), with no local server and no
GPU on your side. Name the endpoint and run the example as before:

```bash
export AWS_REGION=<region>   # the region you deployed to (AWS_DEFAULT_REGION works too)
export STRANDS_DECIDER_SAGEMAKER_ENDPOINT=strands-decider-2b-qwen3-5-v1-2610
python examples/strands/tool_call_intervention.py
```

Only the transport changes. `_client.py` sends the same `/v1/systemone` body through
`sagemaker-runtime` `InvokeEndpoint`, signed with your AWS credentials, so the questions, the
policy and the output are the same as against a local server. In your own code:

```python
from _client import Decider

decider = Decider(sagemaker_endpoint="strands-decider-2b-qwen3-5-v1-2610", region="us-west-2")
answers = decider.ask("Help! My payouts have been failing for 3 days!",
                      {"is_urgent": Decider.noul("Does this convey urgency?")})
```

The region is the `region=` argument if you pass one, else `AWS_REGION`, else
`AWS_DEFAULT_REGION`, else the region of your active AWS profile (`AWS_PROFILE`, `~/.aws/config`).

The calling identity needs `sagemaker:InvokeEndpoint` on the endpoint, and nothing else: the
container has no authentication of its own. The example does not use `describe-endpoint`. Its
start-up check sends one small real request instead, so a wrong name, region or credential fails
there rather than in the middle of an agent run. `STRANDS_DECIDER_URL` is ignored while
`STRANDS_DECIDER_SAGEMAKER_ENDPOINT` is set.

An endpoint bills for every hour it runs. `npx cdk destroy` removes it when you are done.

The scenario is deliberately small. The agent has one `get_weather` tool and a system prompt
that makes it eager, and the user asks "*What's the weather?*" without saying where. The agent
guesses a city and calls the tool anyway. Before that call runs, strands decider reads the
conversation and the proposed call and answers two narrow yes/no questions about it:

1. Are the argument values grounded in facts the user actually provided?
2. Is it premature to call this tool now, before clarifying with the user?

A few lines of plain Python turn those probabilities into a decision, and the agent goes back
to ask which city you meant instead of confidently reporting the weather somewhere nobody
mentioned.

The seam is Strands' intervention system. You write an `InterventionHandler` with a
`before_tool_call` method and pass it to the agent, and it runs before any tool executes. What
you return is a typed action rather than text: the example uses `Proceed` to let the call run
and `Guide` to hand the model back its turn with feedback, which is what makes this different
from a hard allow/deny gate — `Guide` corrects the course instead of refusing. The handler is
an ordinary Python class and Strands has no opinion about what goes inside it, so the same
shape holds whether you call a decision model, a policy engine, or another agent.

Why classify instead of letting the agent check itself? The gate is a local forward pass in a
couple of hundred milliseconds, the conversation never leaves the machine, and each verdict is
a number your `if` statements branch on rather than prose the agent can negotiate with.

> This example is an illustration rather than a recommendation: the questions, the thresholds
> and the policy were all picked by hand. The point is that a decision this cheap can sit in a
> path where an LLM call never could.
