from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Any, Annotated
from pydantic import BaseModel
import gradio as gr

# todo introduce ordering flow using lang graph
# 1) agent suggests buying an item related to the plant's upkeep
# 2) user confirms they would like to buy it or not
# 3) agent needs user to confirm 'secret key' to go ahead with purchase on user's behalf
# 4a) secret key correct: agent goes ahead with the purchase and adds it to purchased items (item, date)
# 4b) secret key incorrect: agent allows 3 attempts for user to get the secret key correct, otherwise the purchase is cancelled
class CustomState(AgentState):
    user_plants: list[str]
    user_details: dict[str, Any]

@tool
def add_user_plants(new_user_plants: list[str], runtime: ToolRuntime):
    """Add plants to the user's inventory."""
    existing_user_plants = runtime.state.get("user_plants")
    # Combine and remove duplicates while preserving order
    combined = list[str](existing_user_plants)
    for plant in new_user_plants:
        if plant not in combined:
            combined.append(plant)

    return Command(update={  
        "user_plants": combined,
        "messages": [
            ToolMessage(
                f"Successfully added {new_user_plants} to user's inventory",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def add_user_details(new_user_details: dict[str, Any], runtime: ToolRuntime):
    """Add details to the user's details."""
    existing_user_details = runtime.state.get("user_details")
    result = dict[str, Any](existing_user_details)
    result.update(new_user_details)

    return Command(update={  
        "user_details": result,
        "messages": [
            ToolMessage(
                f"Successfully added {new_user_details} to user details",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@before_model
def inspect_state(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print('agent_state: ', state)


agent = create_agent(
    model="gpt-5",
    system_prompt="""
    You are a helpful plant assistant. 
    Keep an inventory of the user's plants, provide care feedback.
    Record key user details to help personalise the conversation.
    """,
    middleware=[inspect_state],
    state_schema=CustomState,
    tools=[add_user_details, add_user_plants],
    checkpointer=InMemorySaver()
)

def plant_chat(message, history):
    config = {"configurable": {"thread_id": "1"}}
    
    state_snapshot = agent.get_state(config)
    initial_state = state_snapshot.values if state_snapshot.values else {'user_plants': [], 'user_details': {}}

    print('state: ', initial_state)

    latest_message = ""
    global state
    for chunk in agent.stream(input={"messages": [{"role": "user", "content": message}], 
                                'user_plants': initial_state.get('user_plants', []), 
                                'user_details': initial_state.get('user_details', {})},
                                    config={"configurable": {"thread_id": "1"}}, 
                                    stream_mode="messages"):
        content = ""
        if isinstance(chunk, tuple) and len(chunk) > 0:
            if(getattr(chunk[0], 'type', '') != 'tool'):
                content = getattr(chunk[0], 'content', '')
                if content:
                    latest_message += content
                    yield latest_message

# for debugging with gradio
demo = gr.ChatInterface(fn=plant_chat)
demo.launch()