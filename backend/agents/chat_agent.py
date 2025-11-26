from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Any
from pydantic import BaseModel
# import gradio as gr

# TODO implement a store which stores different user's information
# Pass the user information in as context to the agent based on the current user interacting with the agent

class CustomState(AgentState):
    user_plants: [str]
    user_details: dict

store = InMemoryStore()

state = {
    'user_plants' : [],
    'user_details' : {}
}


@tool
def add_to_calendar(description: str, date: str) -> str:
    """Add a reminder to the user's calendar
    """
    print(f'Added reminder {description} for {date}')
    return "Added to calendar"

@tool
def search_amazon(query: str) -> {}:
    """Search amazon for products the user needs to care for their plants
    """
    print(f'Searching amazon for {query}')
    return {
        'watering can': 1,
        'plant food': 1
    }

# update state of user's plants and user's details
@tool
def add_user_plant(new_user_plant: str):
    """Add a plant to the user's inventory."""
    global state
    state['user_plants'].append(new_user_plant)
    return "Successfully added plant to user's inventory"

@tool
def add_user_detail(detail_key: str, detail_value: str):
    """Add a detail to the user's details."""
    global state
    state['user_details'][detail_key] = detail_value
    return "Successfully added user detail"


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
    tools=[add_user_detail, add_user_plant],
    store=store,
    checkpointer=InMemorySaver()
)

def plant_chat(message, history):
    latest_message = ""
    global state
    for chunk in agent.stream(input={"messages": [{"role": "user", "content": message}], 'user_plants': state['user_plants'], 'user_details': state['user_details']},
                                config={"configurable": {"thread_id": "1"}}, 
                                stream_mode="messages"):
        content = ""
        if isinstance(chunk, tuple) and len(chunk) > 0:
            content = getattr(chunk[0], 'content', '')
            if content:
                latest_message += content
                yield latest_message

# for debugging with gradio
# demo = gr.ChatInterface(fn=plant_chat, type="messages")
# demo.launch()