# case-2-agent-test

**Direct** ("naked") Dispute Ops agent for case 2, for tests: Bedrock, Case Desk and the Card
Network Portal **without proxy-server**. Same code, prompt and tools as `case-2-agent` (the
proxy agent); only the configuration differs. Running both on the same scenario shows what the
proxy changes (`card-network-agent` `plans/dispute-ops/03-tests-secure-vs-naked.md`).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # CASE_DESK_URL/TOKEN, CARD_NETWORK_URL/TOKEN, AWS_PROFILE for Bedrock
CASE_IDS=case_189_unrecognized RUN_ONCE=true python -m dispute_agent
```

The log line `session finished: ... actions:` lists every write the model attempted and its
result (the test trace).

## AWS (ECS)

`one-dev-dispute-agent-direct` service in `one-dev-cluster` (`one-infrastructure/ecs_dispute_agent.tf`),
polling open cases like the proxy agent.

| | |
|---|---|
| Case Desk | `case-desk-test` (Cloud Map DNS, schema `dispute_case_desk_test`), Bearer = test-backend gateway token |
| Card Network | `card-network` on the backend-2 load balancer `:8443` (our CA, trusted in the Dockerfile), Bearer = `MARKETPLACE_API_TOKEN` |
| LLM | Bedrock directly, task role |

Card Network has no test copy: like the marketplace in case 1, the direct agent writes to the same
`dispute_network` data as the proxy agent. Reset before each scenario:
`POST /demo/reset` on card-network (resets `dispute_network` and the prod `dispute_case_desk`) and on
case-desk-test (resets `dispute_case_desk_test`). The agent handles each open case once per task;
after a reset, restart the service (new deployment) to run the cases again.

CI (`.github/workflows/deploy.yml`, manual): image to ECR `one-dev-dispute-agent-direct`, new
deployment of the service.
