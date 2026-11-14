"""End-to-end test of the agent tree with a scripted model.

No API key needed. A ScriptedLlm replaces Gemini and returns a fixed list of
tool calls per agent. Everything else is real ADK: the runner, agent transfer,
tool execution, session state, and {placeholder} injection in instructions.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from order_assistant_agent import agent as app  # noqa: E402


class ScriptedLlm(BaseLlm):
    """Returns the next scripted part each time the agent calls the model."""

    model: str = "scripted"
    script: list[Any] = []
    instructions: list[str] = []

    async def generate_content_async(self, llm_request, stream=False):
        self.instructions.append(str(llm_request.config.system_instruction))
        part = self.script.pop(0)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def call(tool: str, /, **args) -> types.Part:
    return types.Part.from_function_call(name=tool, args=args)


def say(text: str) -> types.Part:
    return types.Part(text=text)


def script_agents(taste: str) -> dict[str, ScriptedLlm]:
    scripts = {
        "root": ScriptedLlm(script=[
            call("save_profile", name="Priya", taste=taste),
            call("transfer_to_agent", agent_name="catalog_agent"),
        ]),
        "catalog": ScriptedLlm(script=[
            call("show_menu"),
            call("record_view", item="Spicy Chicken Burger"),
            call("add_item", item="Spicy Chicken Burger"),
            call("transfer_to_agent", agent_name="recommender_agent"),
        ]),
        "recommender": ScriptedLlm(script=[
            call("recommend"),
            call("transfer_to_agent", agent_name="order_summary_agent"),
        ]),
        "summary": ScriptedLlm(script=[say("Order summary written.")]),
    }
    app.root_agent.model = scripts["root"]
    app.catalog_agent.model = scripts["catalog"]
    app.recommender_agent.model = scripts["recommender"]
    app.order_summary_agent.model = scripts["summary"]
    return scripts


async def run_conversation() -> tuple[dict, list[tuple[str, str]]]:
    runner = InMemoryRunner(agent=app.root_agent, app_name="test_app")
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="u1"
    )
    message = types.Content(
        role="user", parts=[types.Part(text="Hi, I'm Priya. I want a burger.")]
    )
    authors_and_text = []
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    authors_and_text.append((event.author, part.text))
    session = await runner.session_service.get_session(
        app_name="test_app", user_id="u1", session_id=session.id
    )
    return dict(session.state), authors_and_text


@pytest.fixture(autouse=True)
def restore_models():
    agents = [
        app.root_agent,
        app.catalog_agent,
        app.recommender_agent,
        app.order_summary_agent,
    ]
    originals = [(a, a.model) for a in agents]
    yield
    for agent, model in originals:
        agent.model = model


def test_full_flow_with_taste():
    scripts = script_agents(taste="spicy")
    state, replies = asyncio.run(run_conversation())

    # Every agent handed off and wrote to shared state.
    assert state["name"] == "Priya"
    assert state["taste"] == "spicy"
    assert state["viewed"] == ["Spicy Chicken Burger"]
    assert state["cart"] == ["Spicy Chicken Burger"]
    assert state["suggested"] == ["Fries", "Cola"]

    # The summary agent ran last and its instruction was filled from state.
    assert replies[-1] == ("order_summary_agent", "Order summary written.")
    rendered = scripts["summary"].instructions[0]
    assert "Name: Priya" in rendered
    assert "Taste preference: spicy" in rendered
    assert "Order: ['Spicy Chicken Burger']" in rendered
    assert "Suggested but not added: ['Fries', 'Cola']" in rendered

    # Nothing was left unused in any script: each agent made the calls we expect.
    assert all(not s.script for s in scripts.values())


def test_optional_taste_is_skipped():
    scripts = script_agents(taste="")
    state, _ = asyncio.run(run_conversation())

    assert "taste" not in state
    rendered = scripts["summary"].instructions[0]
    assert "Taste preference: \n" in rendered
    assert "Name: Priya" in rendered


def test_unknown_item_is_rejected():
    class Ctx:
        state: dict = {}

    ctx = Ctx()
    ctx.state = {}
    assert app.add_item("Mystery Dish", ctx)["status"] == "not_found"
    assert "cart" not in ctx.state
