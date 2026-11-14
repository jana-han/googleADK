"""Order assistant built with Google ADK.

Four agents share one session state:
  order_assistant (root) -> catalog_agent, recommender_agent, order_summary_agent
"""

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Mock data. In production, replace these with your catalog and your
# recommendation service.
# ---------------------------------------------------------------------------

MENU = {
    "Spicy Chicken Burger": 9.50,
    "Veggie Wrap": 8.00,
    "Fries": 3.00,
    "Cola": 2.50,
    "Iced Tea": 2.80,
}

# A lookup table standing in for a real recommendation model.
PAIRS = {
    "Spicy Chicken Burger": ["Fries", "Cola"],
    "Veggie Wrap": ["Iced Tea"],
    "Fries": ["Cola"],
}


# ---------------------------------------------------------------------------
# Tools. The docstring and type hints are what the model reads, so keep them
# clear. Every tool returns a dict.
# ---------------------------------------------------------------------------


def show_menu() -> dict:
    """Returns every item that can be ordered, with its price.

    Returns:
        A dict with a status and an items mapping of name to price.
    """
    return {"status": "success", "items": MENU}


def save_profile(name: str, taste: str, tool_context: ToolContext) -> dict:
    """Saves the user's name and optional taste preference.

    Args:
        name: The user's name.
        taste: A taste preference such as "spicy". Pass an empty string if none.

    Returns:
        A dict with a status.
    """
    tool_context.state["name"] = name
    if taste:
        tool_context.state["taste"] = taste
    return {"status": "saved"}


def record_view(item: str, tool_context: ToolContext) -> dict:
    """Records that the user looked at an item.

    Args:
        item: The exact item name from the menu.

    Returns:
        A dict with a status and the item.
    """
    viewed = tool_context.state.get("viewed", [])
    # Assign a new list so the change is tracked in session state.
    tool_context.state["viewed"] = viewed + [item]
    return {"status": "recorded", "item": item}


def add_item(item: str, tool_context: ToolContext) -> dict:
    """Adds an item to the user's order.

    Args:
        item: The exact item name from the menu.

    Returns:
        A dict with a status and the item.
    """
    if item not in MENU:
        return {"status": "not_found", "item": item}
    cart = tool_context.state.get("cart", [])
    tool_context.state["cart"] = cart + [item]
    return {"status": "added", "item": item}


def recommend(tool_context: ToolContext) -> dict:
    """Suggests add-ons based on the items already in the order.

    Returns:
        A dict with a status and a list of suggested item names.
    """
    cart = tool_context.state.get("cart", [])
    suggestions = []
    for item in cart:
        for other in PAIRS.get(item, []):
            if other not in cart and other not in suggestions:
                suggestions.append(other)
    tool_context.state["suggested"] = suggestions
    return {"status": "ok", "suggestions": suggestions}


# ---------------------------------------------------------------------------
# Sub-agents. Descriptions are what the parent reads to decide who to hand off
# to, so write them like a job posting.
# ---------------------------------------------------------------------------

catalog_agent = LlmAgent(
    name="catalog_agent",
    model=MODEL,
    description=(
        "Shows the menu, records items the user looks at, "
        "and adds chosen items to the order."
    ),
    instruction=(
        "Call show_menu to list items. When the user asks about an item, "
        "call record_view. When they choose one, call add_item."
    ),
    tools=[show_menu, record_view, add_item],
)

recommender_agent = LlmAgent(
    name="recommender_agent",
    model=MODEL,
    description="Suggests add-ons once the user has at least one item in the order.",
    instruction="Call recommend and present the suggestions. Do not invent items.",
    tools=[recommend],
)

# No tools on purpose. It only reads state through the {placeholders} below.
# {taste?} and {suggested?} are optional: the agent still runs if they are
# missing from state.
order_summary_agent = LlmAgent(
    name="order_summary_agent",
    model=MODEL,
    description="Writes the final order summary once the user is done choosing.",
    instruction=(
        "Write a short order summary.\n"
        "Name: {name}\n"
        "Taste preference: {taste?}\n"
        "Order: {cart}\n"
        "Suggested but not added: {suggested?}\n"
        "Do not add items the user did not choose."
    ),
)

# ---------------------------------------------------------------------------
# Root agent. ADK looks for a variable named root_agent.
# ---------------------------------------------------------------------------

root_agent = LlmAgent(
    name="order_assistant",
    model=MODEL,
    description="Front door for an ordering assistant.",
    instruction=(
        "Greet the user. Ask their name and any taste preference, then "
        "call save_profile. Hand off to the right specialist after that. "
        "Suggest add-ons only through the recommender, never from your own guess."
    ),
    tools=[save_profile],
    sub_agents=[catalog_agent, recommender_agent, order_summary_agent],
)
