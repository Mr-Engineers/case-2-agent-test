"""Main loop: scan Case Desk for open cases, run one dispute session per case in parallel."""

import asyncio
import logging
import time

from .apps import AppsClient
from .config import Settings
from .llm import Llm
from .session import SessionResult, run_case

log = logging.getLogger("dispute_agent")


async def scan(apps: AppsClient, settings: Settings) -> list[str]:
    if settings.selected_case_ids:
        return settings.selected_case_ids
    session_id = await apps.create_session(task="Scan Case Desk for open dispute cases")
    try:
        outcome = await apps.call(session_id, "case_desk", "GET", "/v1/cases", params={"status": "open"})
    finally:
        await apps.close_session(session_id)
    if outcome.status != "ok":
        log.error("open cases scan failed: %s %s", outcome.status, outcome.body)
        return []
    return [case["id"] for case in outcome.body.get("cases", [])]


async def main() -> None:
    settings = Settings()
    log.info(
        "mode=%s llm=%s case_desk=%s card_network=%s",
        settings.agent_mode, settings.llm_base_url, settings.case_desk_url, settings.card_network_url,
    )
    apps = AppsClient(settings)
    llm = Llm(settings)
    semaphore = asyncio.Semaphore(settings.max_parallel_sessions)
    in_flight: dict[str, asyncio.Task[SessionResult]] = {}
    # A case stays open after a refund, so a finished case is not picked up again by this process
    handled: set[str] = set()
    cooldown_until: dict[str, float] = {}

    async def resolve(case_id: str) -> SessionResult:
        async with semaphore:
            try:
                result = await run_case(apps, llm, settings, case_id)
            except Exception:
                log.exception("[%s] session failed", case_id)
                result = SessionResult(case_id, None, False, "error")
        if result.finished:
            handled.add(case_id)
        else:
            cooldown_until[case_id] = time.monotonic() + settings.case_retry_cooldown_s
        actions = ", ".join(f"{action.tool}={action.status}" for action in result.actions) or "none"
        log.info("[%s] session finished: finished=%s, actions: %s, %s", case_id, result.finished, actions, result.summary)
        return result

    try:
        while True:
            try:
                case_ids = await scan(apps, settings)
            except Exception:
                log.exception("scan failed")
                case_ids = []

            now = time.monotonic()
            for case_id in case_ids:
                if case_id in in_flight or case_id in handled or cooldown_until.get(case_id, 0) > now:
                    continue
                task = asyncio.create_task(resolve(case_id))
                in_flight[case_id] = task
                task.add_done_callback(lambda _, case_id=case_id: in_flight.pop(case_id, None))
            log.info("scan: %d open case(s), %d session(s) running", len(case_ids), len(in_flight))

            if settings.run_once:
                await asyncio.gather(*in_flight.values())
                return
            await asyncio.sleep(settings.poll_interval_s)
    finally:
        await apps.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
