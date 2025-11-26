from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Any, List
from langchain_core.runnables import RunnableLambda
from langchain_core.globals import set_debug
import gradio as gr

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    user_plants: List[str] = Field([], description="The user's list of plants. Add to the list, do not replace the list.")


def RExtract(pydantic_class, llm, prompt):
    parser = PydanticOutputParser[Any](pydantic_object=pydantic_class)
    populated_prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return populated_prompt | llm | parser


instruct_llm = init_chat_model(model="openai:gpt-4o-mini")

internal_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful plant assistant. Update the knowledge base {format_instructions}."),
        ("user", "CURRENT KNOWLEDGE BASE: {knowledge_base}\n User: {input}")
    ]
)

external_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful plant assistant, helping users look after their plants. Your running knowledge base is {knowledge_base}"),
        ("assistant", "{output}"),
        ("user", "{input}")
    ]
)

internal_chain = RExtract(KnowledgeBase, instruct_llm, internal_prompt)
external_chain = external_prompt | instruct_llm | StrOutputParser()

know_base = KnowledgeBase()

def plant_chat(message, history):
    global know_base
    know_base = internal_chain.invoke({'input': message, 'knowledge_base': know_base})

    output = "" if not history else history[-1]['content']
    buffer = ""
    for token in external_chain.stream({'input': message, 'output': output, 'knowledge_base': know_base}):
        buffer += token
        yield buffer

demo = gr.ChatInterface(fn=plant_chat, type="messages")
demo.launch()