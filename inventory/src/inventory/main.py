from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Get API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")

# Disable tracing
set_tracing_disabled(disabled=True)

# External client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Chat Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Inventory store
inventory = {}

# ------------------- TOOLS -------------------

@function_tool
async def add_item(name: str, quantity: int) -> str:
    """Add item with given quantity to inventory."""
    inventory[name] = inventory.get(name, 0) + quantity
    return f"✅ Added {quantity} x {name}"


@function_tool
async def delete_item(name: str, quantity: int = None) -> str:
    """Delete an item or reduce its quantity from inventory."""
    if name not in inventory:
        return f"⚠️ {name} not found in inventory."

    if quantity is None or quantity >= inventory[name]:
        del inventory[name]
        return f"🗑️ Deleted item {name}"
    else:
        inventory[name] -= quantity
        return f"➖ Removed {quantity} x {name} (Remaining: {inventory[name]})"


@function_tool
async def list_inventory() -> str:
    """List all inventory items with quantities."""
    if not inventory:
        return "📦 Inventory is empty."
    return "\n".join([f"- {item} : {qty}" for item, qty in inventory.items()])


# ------------------- AGENT -------------------

agent = Agent(
    name="Inventory Manager",
    instructions="""
    You are an Inventory Manager. 
    Use tools to add, delete, or list inventory items. 

    ✅ Whenever you add or delete an item, ALWAYS call 'list_inventory' afterwards.  

    In your final output, clearly mention:
    - What action was performed (added / deleted / updated).
    - The updated inventory list with item names and quantities.  

    🔹 Keep item names exactly as they were added (e.g., 'HP Laptop', 'Dell Laptop').
    """,
    tools=[add_item, delete_item, list_inventory],
    model=model,
)

# ------------------- MAIN -------------------

async def main(user_input: str):
    result = await Runner.run(agent, input=user_input)
    print("\n==== FINAL RESULT ====")
    print(result.final_output)
    print("======================\n")


def start():
    # Example prompt
    asyncio.run(main("Add 5 HP Laptops & 3 Iphone in the inventory. Then delete 2 HP Laptops & 1 Iphone. Finally, list the inventory."))