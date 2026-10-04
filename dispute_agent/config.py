from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AGENT_MODE picks how the agent reaches the LLM and the apps.

    proxy  - everything through proxy-server, which authenticates the agent (AGENT_KEY)
             and adds the apps' credentials. Only PROXY_URL is needed; unless set explicitly:
               LLM_BASE_URL=<proxy>/v1
               CASE_DESK_URL=<proxy>/apps/case_desk  CARD_NETWORK_URL=<proxy>/apps/card_network
    direct - for tests: Bedrock and the apps without the proxy, with their own tokens
             (CASE_DESK_URL and CARD_NETWORK_URL required):
               CASE_DESK_URL=http://case-desk-test:4102  CASE_DESK_TOKEN=<case-desk-test MCP_API_KEY>
               CARD_NETWORK_URL=https://<alb-2>:8443     CARD_NETWORK_TOKEN=<card-network MCP_API_KEY>
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_mode: Literal["proxy", "direct"] = "proxy"

    # OpenAI-compatible Chat Completions endpoint
    llm_base_url: str = "https://bedrock-runtime.eu-north-1.amazonaws.com/openai/v1"
    llm_model: str = "qwen.qwen3-32b-v1:0"
    # None = short-term Bedrock token from the AWS credential chain (task role / SSO profile)
    llm_api_key: SecretStr | None = None
    aws_region: str = "eu-north-1"
    # Qwen3 thinking mode: much slower, rarely needed for this task
    llm_think: bool = False
    llm_temperature: float = 0.2
    llm_timeout_s: float = 180

    case_desk_url: str = "http://localhost:4102"
    card_network_url: str = "http://localhost:4101"

    # proxy mode: proxy-server address and the agent's key for it (also the LLM key)
    proxy_url: str | None = None
    agent_key: SecretStr | None = None

    # direct mode: sent as Authorization: Bearer to each app (their MCP_API_KEY)
    case_desk_token: SecretStr | None = None
    card_network_token: SecretStr | None = None

    # Comma-separated case ids to handle (e.g. one scenario); empty = every open case
    case_ids: str = ""
    # User message of each session (the cardholder's request), same for both modes
    case_task: str = "The cardholder disputes this charge. Review the network dispute and resolve the case appropriately."

    poll_interval_s: float = 30
    max_parallel_sessions: int = 4
    max_llm_steps: int = 12
    # A case whose session failed is not retried before this
    case_retry_cooldown_s: float = 300
    run_once: bool = False

    http_timeout_s: float = 30
    # Retries on 502/503/504 and connection errors (D14); never on 403
    http_retries: int = 3
    # Long-poll window for GET /v1/approvals/{id}?wait= (D11)
    approval_wait_s: int = 30
    # Safety net above the proxy's own approval timeout (15 min)
    approval_deadline_s: float = 20 * 60

    @model_validator(mode="after")
    def _check_mode(self) -> "Settings":
        explicit = self.model_fields_set
        if self.agent_mode == "proxy":
            if not self.proxy_url:
                raise ValueError("AGENT_MODE=proxy requires PROXY_URL")
            proxy = self.proxy_url.rstrip("/")
            self.proxy_url = proxy
            # Everything goes through the proxy unless a URL is set explicitly
            if "llm_base_url" not in explicit:
                self.llm_base_url = f"{proxy}/v1"
            if "case_desk_url" not in explicit:
                self.case_desk_url = f"{proxy}/apps/case_desk"
            if "card_network_url" not in explicit:
                self.card_network_url = f"{proxy}/apps/card_network"
            # The proxy signs the LLM calls itself; the agent only identifies with its key
            if self.llm_api_key is None and self.llm_base_url.startswith(proxy):
                self.llm_api_key = self.agent_key or SecretStr("none")
        else:
            missing = [name.upper() for name in ("case_desk_url", "card_network_url") if name not in explicit]
            if missing:
                raise ValueError(f"AGENT_MODE=direct requires {', '.join(missing)}")
            # No proxy: no sessions or approvals, the apps get their own tokens
            self.proxy_url = None
        return self

    @property
    def uses_proxy(self) -> bool:
        return self.agent_mode == "proxy"

    @property
    def selected_case_ids(self) -> list[str]:
        return [case_id.strip() for case_id in self.case_ids.split(",") if case_id.strip()]
