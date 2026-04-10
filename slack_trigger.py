#!/usr/bin/env python3
"""
Slack Trigger for Autonomous Coding Agent
==========================================

This file is the Slack-facing input layer for the autonomous agent harness.
It listens for messages via Socket Mode — which opens an outbound WebSocket to
Slack's servers, so no inbound port or public URL is required. This makes it
ideal for a local Mac Mini behind a home router or corporate firewall.

Architecture overview:
    Slack message
        → slack_trigger.py  (this file: parse + ack)
            → run_autonomous_agent()  (agent.py: the same loop the CLI demo uses)
                → create_client()  (client.py: Claude SDK + Arcade MCP gateway)
                    → Orchestrator prompt + AGENT_DEFINITIONS
                        → Linear / Coding / GitHub / Slack sub-agents

The trigger is purely an input layer. All orchestration, tool calls, Linear
issue management, GitHub PR creation, and Slack progress notifications are
handled by the existing harness exactly as the CLI demo does them.

Message format:
    project-name: task description
    e.g.  spendlog: fix the category filter crash on Android only on rotation

If no "project-name:" prefix is given the DEFAULT_PROJECT env var is used,
matching the --project-dir default behaviour of autonomous_agent_demo.py.

Usage:
    uv run python slack_trigger.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing modules that read env vars at import time.
# arcade_config.py reads ARCADE_API_KEY and ARCADE_GATEWAY_SLUG at module
# level, so load_dotenv() must come first.
load_dotenv()

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

# Import the harness entry point — this is the exact same coroutine that
# autonomous_agent_demo.py schedules via asyncio.run(). It handles the full
# agent loop: project directory setup, Linear initialisation check, session
# creation via create_client(), prompt selection, orchestrator delegation,
# retry logic, and the PROJECT_COMPLETE shutdown signal.
from agent import run_autonomous_agent

# Available models — kept in sync with autonomous_agent_demo.py so that
# ORCHESTRATOR_MODEL env var has the same effect in both entry points.
AVAILABLE_MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-5-20251101",
}


# =============================================================================
# Environment variables
# =============================================================================

# xoxb- bot token — from your Slack app's "OAuth & Permissions" page.
# Requires at minimum: channels:history, channels:read, chat:write, app_mentions:read
SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN", "")

# xapp- Socket Mode app-level token — from "App Level Tokens" in your app settings.
# Must have the connections:write scope, which is required for Socket Mode.
SLACK_APP_TOKEN: str = os.environ.get("SLACK_APP_TOKEN", "")

# Fallback project directory name used when a message has no "project:" prefix.
# Maps to the same concept as --project-dir in autonomous_agent_demo.py.
DEFAULT_PROJECT: str = os.environ.get("DEFAULT_PROJECT", "")

# Orchestrator model selection — same env var used by autonomous_agent_demo.py.
# The orchestrator just delegates, so haiku is cost-effective and sufficient.
_model_key: str = os.environ.get("ORCHESTRATOR_MODEL", "haiku").lower()
ORCHESTRATOR_MODEL_KEY: str = _model_key if _model_key in AVAILABLE_MODELS else "haiku"

# Base directory where project subdirectories are created.
# Mirrors the --generations-base flag / GENERATIONS_BASE_PATH env var in the CLI demo.
DEFAULT_GENERATIONS_BASE: Path = Path(
    os.environ.get("GENERATIONS_BASE_PATH", "./generations")
)

# Optional channel filter. When set, the trigger only acts on messages in this
# channel. Use the channel ID (e.g. C01ABCDEF) rather than the display name,
# because Socket Mode events carry channel IDs, not names. The existing
# SLACK_CHANNEL env var (used by the Slack agent for outgoing notifications)
# stores a display name, so you may need separate values for each purpose.
SLACK_CHANNEL_FILTER: str = os.environ.get("SLACK_CHANNEL", "")


# =============================================================================
# Slack App
# =============================================================================

# AsyncApp + AsyncSocketModeHandler give us a fully async Slack client.
# This means the message handler runs inside the same asyncio event loop
# as the agent, allowing asyncio.create_task() to schedule agent sessions
# concurrently without blocking the Slack connection.
app = AsyncApp(token=SLACK_BOT_TOKEN)


# =============================================================================
# Helpers
# =============================================================================

def parse_message(text: str) -> tuple[str | None, str]:
    """
    Parse a Slack message into (project_name, task_description).

    Expected format:  project-name: task description
    If no recognisable "name:" prefix is found, returns (None, full_text).
    The caller substitutes DEFAULT_PROJECT in that case.

    A prefix is only treated as a project name if it contains no spaces,
    preventing false positives on sentences like "Fix: the login is broken".
    URL schemes ("http", "https", "ftp") are also excluded.

    Examples:
        "spendlog: fix crash on rotation"  → ("spendlog", "fix crash on rotation")
        "fix crash on rotation"            → (None, "fix crash on rotation")
        "my app: add dark mode"            → (None, "my app: add dark mode")  # space in prefix
        "http://example.com"               → (None, "http://example.com")
    """
    if ":" not in text:
        return None, text.strip()

    prefix, _, rest = text.partition(":")
    stripped = prefix.strip()

    # Reject multi-word prefixes and URL schemes
    if not stripped or " " in stripped or stripped.lower() in ("http", "https", "ftp"):
        return None, text.strip()

    return stripped, rest.strip()


def resolve_project_dir(project_name: str) -> Path:
    """
    Resolve a project name to an absolute Path.

    Mirrors the path-resolution logic in autonomous_agent_demo.py:
    - Absolute paths are used as-is.
    - Relative names are placed inside GENERATIONS_BASE_PATH.

    Args:
        project_name: A project directory name (e.g. "spendlog") or absolute path.

    Returns:
        Absolute Path to the project directory (may not exist yet; the harness
        creates it via project_dir.mkdir(parents=True, exist_ok=True) in
        run_autonomous_agent()).
    """
    project_path = Path(project_name)
    if project_path.is_absolute():
        return project_path

    generations_base = DEFAULT_GENERATIONS_BASE
    if not generations_base.is_absolute():
        generations_base = Path.cwd() / generations_base

    # Strip leading "./" for cleaner paths — same normalisation as the CLI demo
    clean_name = project_name.lstrip("./")
    return generations_base / clean_name


# =============================================================================
# Message handler
# =============================================================================

@app.event("message")
async def handle_message(event: dict, say) -> None:
    """
    Handle incoming Slack messages and trigger the agent harness.

    Flow:
      1. Filter out bot messages, edits, deletions, and wrong channels.
      2. Parse the "project-name: task description" format.
      3. Reply in the thread immediately to acknowledge receipt.
      4. Schedule run_autonomous_agent() as an asyncio task so the handler
         returns quickly — Socket Mode requires a fast acknowledgement.

    The asyncio task runs concurrently in the same event loop as the Slack app.
    The existing harness (agent.py → client.py → orchestrator) takes over from
    there, including posting progress updates via the Slack sub-agent.

    Args:
        event: Raw Slack event dict (provided by slack-bolt).
        say:   Slack response helper — posts to the same channel as the event.
    """
    # --- Ignore bot messages to prevent feedback loops ---
    # The existing Slack sub-agent posts progress notifications; without this
    # guard, each of those would re-trigger the harness indefinitely.
    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        return

    # Ignore non-message subtypes (channel_join, message_changed, etc.)
    if "subtype" in event:
        return

    # --- Optional channel filter ---
    # Only applies when SLACK_CHANNEL is set. Useful when the bot is invited
    # to multiple channels but should only accept tasks from one.
    if SLACK_CHANNEL_FILTER:
        event_channel = event.get("channel", "")
        if event_channel != SLACK_CHANNEL_FILTER:
            return

    text: str = event.get("text", "").strip()
    if not text:
        return

    # --- Parse "project-name: task description" ---
    project_name, task = parse_message(text)

    if not project_name:
        if DEFAULT_PROJECT:
            project_name = DEFAULT_PROJECT
        else:
            # No project name in message and no fallback configured
            await say(
                text=(
                    "Please specify a project using the format:\n"
                    "`project-name: task description`\n\n"
                    "Or set `DEFAULT_PROJECT` in the bot's environment to use a fallback."
                ),
                thread_ts=event.get("ts"),
            )
            return

    if not task:
        await say(
            text=(
                f"Got project `{project_name}` but no task description. "
                "Please describe what you want done."
            ),
            thread_ts=event.get("ts"),
        )
        return

    # --- Resolve project directory ---
    # Uses the same path logic as autonomous_agent_demo.py: relative names go
    # into GENERATIONS_BASE_PATH, absolute paths are used directly.
    project_dir: Path = resolve_project_dir(project_name)
    model_id: str = AVAILABLE_MODELS[ORCHESTRATOR_MODEL_KEY]

    # --- Acknowledge in thread immediately ---
    # This lets the user know work has started before the agent session begins.
    # Further progress updates come from the harness's Slack sub-agent, which
    # posts to the configured SLACK_CHANNEL as it completes each Linear issue.
    await say(
        text=(
            f":robot_face: Starting agent for *{project_name}*\n"
            f"*Task:* {task}\n"
            f"*Project dir:* `{project_dir}`\n"
            f"*Model:* {ORCHESTRATOR_MODEL_KEY}\n\n"
            "Progress updates will follow as the agent works through Linear issues."
        ),
        thread_ts=event.get("ts"),
    )

    print(f"\n[Slack trigger] New task received")
    print(f"  Project : {project_name}")
    print(f"  Task    : {task}")
    print(f"  Dir     : {project_dir}")
    print(f"  Model   : {ORCHESTRATOR_MODEL_KEY}")
    print()

    # --- Launch the agent harness as a concurrent asyncio task ---
    #
    # asyncio.create_task() schedules run_autonomous_agent() to run in the
    # background within the same event loop that drives the Slack app. The
    # handler returns immediately, satisfying Socket Mode's fast-ack requirement,
    # while the agent session proceeds asynchronously.
    #
    # run_autonomous_agent() is the exact same coroutine autonomous_agent_demo.py
    # calls via asyncio.run(). It handles:
    #   - Creating the project directory
    #   - Checking for .linear_project.json (fresh vs. continuation)
    #   - Creating a new ClaudeSDKClient via create_client() each iteration
    #     (this re-initialises the Arcade MCP gateway, security settings, and
    #      orchestrator system prompt — see client.py for details)
    #   - Sending get_initializer_task() or get_continuation_task() to the agent
    #   - Delegating to Linear / Coding / GitHub / Slack sub-agents
    #   - Retrying on error, stopping on PROJECT_COMPLETE signal
    asyncio.create_task(
        run_autonomous_agent(
            project_dir=project_dir,
            model=model_id,
            max_iterations=None,  # Run until PROJECT_COMPLETE — same as CLI default
        ),
        name=f"agent-{project_name}",
    )


# =============================================================================
# Startup validation
# =============================================================================

def validate_env() -> bool:
    """
    Validate that all required environment variables are set before starting.

    Checks both Slack credentials and the Arcade credentials that the harness
    needs (agent.py → create_client() → validate_arcade_config()).

    Returns:
        True if the configuration is valid, False otherwise.
    """
    errors: list[str] = []

    # Slack credentials
    if not SLACK_BOT_TOKEN:
        errors.append(
            "SLACK_BOT_TOKEN is not set\n"
            "    → Create a Slack app at api.slack.com/apps, enable OAuth,\n"
            "      add bot scopes, install the app, and copy the xoxb- token."
        )
    elif not SLACK_BOT_TOKEN.startswith("xoxb-"):
        errors.append("SLACK_BOT_TOKEN looks invalid — it should start with 'xoxb-'")

    if not SLACK_APP_TOKEN:
        errors.append(
            "SLACK_APP_TOKEN is not set\n"
            "    → In your Slack app settings, go to 'App Level Tokens',\n"
            "      create a token with the connections:write scope, and copy the xapp- token."
        )
    elif not SLACK_APP_TOKEN.startswith("xapp-"):
        errors.append("SLACK_APP_TOKEN looks invalid — it should start with 'xapp-'")

    # Arcade credentials (needed by the harness, checked early to fail fast)
    arcade_key = os.environ.get("ARCADE_API_KEY", "")
    if not arcade_key:
        errors.append(
            "ARCADE_API_KEY is not set\n"
            "    → Get your key from https://api.arcade.dev/dashboard/api-keys"
        )

    arcade_slug = os.environ.get("ARCADE_GATEWAY_SLUG", "")
    if not arcade_slug:
        errors.append(
            "ARCADE_GATEWAY_SLUG is not set\n"
            "    → Create a gateway at https://api.arcade.dev/dashboard/mcp-gateways"
        )

    if errors:
        print("Error: Missing or invalid environment variables:\n")
        for err in errors:
            print(f"  • {err}\n")
        print("Copy .env.example to .env and fill in all required values.")
        return False

    if not DEFAULT_PROJECT:
        print(
            "Warning: DEFAULT_PROJECT is not set.\n"
            "         Messages without a 'project-name:' prefix will prompt the user\n"
            "         to include one. Set DEFAULT_PROJECT to accept bare task messages.\n"
        )

    return True


# =============================================================================
# Entry point
# =============================================================================

async def main() -> None:
    """
    Start the Slack Socket Mode listener.

    AsyncSocketModeHandler establishes a WebSocket connection to Slack and
    keeps it alive, dispatching events to the handlers registered on `app`.
    It handles reconnection automatically if the connection drops, which makes
    it robust for a long-running process on a local machine.
    """
    print("\n" + "=" * 70)
    print("  SLACK TRIGGER — Autonomous Coding Agent")
    print("=" * 70)
    print(f"\nListening via Socket Mode (no public URL required)")
    print(f"Default project : {DEFAULT_PROJECT or '(not set — project prefix required)'}")
    print(f"Generations base: {DEFAULT_GENERATIONS_BASE.resolve()}")
    print(f"Orchestrator    : {ORCHESTRATOR_MODEL_KEY}")
    if SLACK_CHANNEL_FILTER:
        print(f"Channel filter  : {SLACK_CHANNEL_FILTER}")
    else:
        print(f"Channel filter  : none (responding in all channels the bot is in)")
    print()
    print("Message format  : project-name: task description")
    print("Example         : spendlog: fix the category filter crash on rotation")
    print()
    print("Press Ctrl+C to stop.\n")

    # AsyncSocketModeHandler wraps the AsyncApp and manages the Socket Mode
    # WebSocket. start_async() blocks until the process is killed.
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    if not validate_env():
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(0)
