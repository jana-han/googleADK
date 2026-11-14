"""Run the order assistant from Python, without the web UI.

Usage:
    python run_agent.py "Hi, I'm Priya. I like spicy food. What do you have?"

Needs GOOGLE_API_KEY in order_assistant_agent/.env (see .env.example).
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(Path(__file__).parent / "order_assistant_agent" / ".env")

from order_assistant_agent.agent import root_agent  # noqa: E402

APP_NAME = "order_assistant_agent"
USER_ID = "demo_user"


async def chat(messages: list[str]) -> dict:
    """Sends each message in turn and prints the agent's replies.

    Returns the final session state so you can see what the agents saved.
    """
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )

    for text in messages:
        print(f"\nUser: {text}")
        message = types.Content(role="user", parts=[types.Part(text=text)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{event.author}] {part.text}")

    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id
    )
    return dict(session.state)


if __name__ == "__main__":
    prompts = sys.argv[1:] or [
        "Hi, I'm Priya. I like spicy food. What do you have?",
        "Tell me about the Spicy Chicken Burger, then add it to my order.",
        "What goes well with it?",
        "That's all. Please summarise my order.",
    ]
    final_state = asyncio.run(chat(prompts))
    print("\nSession state:", final_state)
