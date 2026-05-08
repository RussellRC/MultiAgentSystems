from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import Literal

# 1. Define the State Structure
class OrderState(BaseModel):
    order_id: int
    items: list[str]
    status: Literal['PENDING', 'VALIDATED', 'COMPLETED', 'FAILED'] = 'PENDING'
    shipping_address: str | None = None

# 2. Define the Agent with Dependency (State)
# The agent operates on OrderState, taking it as input and updating it
agent = Agent[OrderState, str](
    'openai:gpt-4o',
    deps_type=OrderState,
    result_type=str,
)

# 3. Define State Transition Tools
@agent.tool
async def validate_order(ctx: RunContext[OrderState]) -> str:
    if not ctx.deps.items:
        ctx.deps.status = 'FAILED'
        return "Order failed: No items."
    ctx.deps.status = 'VALIDATED'
    return "Order validated successfully."

@agent.tool
async def process_shipping(ctx: RunContext[OrderState], address: str) -> str:
    if ctx.deps.status != 'VALIDATED':
        return "Cannot ship: Order not validated."
    ctx.deps.shipping_address = address
    ctx.deps.status = 'COMPLETED'
    return f"Order shipped to {address}."

# 4. Running the Agent
async def run_order_process():
    # Initial state
    state = OrderState(order_id=1, items=["laptop", "mouse"])

    # Process through steps (normally handled by LLM planning)
    print(f"Initial State: {state.status}")

    # Simulate the agent deciding to validate
    await agent.run("Validate this order.", deps=state)
    print(f"Post-Validation: {state.status}")

    # Simulate the agent deciding to ship
    await agent.run("Ship to 123 AI Lane.", deps=state)
    print(f"Final State: {state.status}, Address: {state.shipping_address}")

# Run the example (requires async environment)
# await run_order_process()
