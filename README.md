# Order assistant (Google ADK)

A recommendation agent: four agents sharing one session state.

```
order_assistant (root)
├── catalog_agent         show_menu, record_view, add_item
├── recommender_agent     recommend
└── order_summary_agent   no tools, reads state through {placeholders}
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp order_assistant_agent/.env.example order_assistant_agent/.env
# edit .env and add your Gemini API key from Google AI Studio

adk web            # web UI: pick "order_assistant_agent"
python run_agent.py   # or run a scripted conversation from Python
```

Run both from this folder (the one that contains `order_assistant_agent/`).

Try: "Hi, I'm Priya. I like spicy food. What do you have?"
Then pick the Spicy Chicken Burger and ask what goes with it.

## Test it (no API key needed)

```bash
pip install pytest
python -m pytest tests -q
```

The tests replace Gemini with a scripted model. Everything else is real ADK:
agent transfer, tool calls, session state and `{placeholder}` injection.

Keep `.env` out of Git.
