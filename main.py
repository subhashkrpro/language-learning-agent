import asyncio
import os
from typing import TypedDict, Optional, Annotated

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Import ONLY local tools
from agent.tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
    create_anki_stack,  # <--- The new batch tool
    get_translation_model  # Helper to reuse LLM logic
)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    target_language: Optional[str]


# Define the tools list explicitly
# We are NOT using MCP anymore to prevent loops
tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
    create_anki_stack
]


def assistant(state: AgentState):
    # Get the LLM using your factory
    llm = get_translation_model()  # Reusing the factory from tools.py

    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content="""
    You are a helpful language learning assistant.

    Your goal is to:
    1. Get random words for the user.
    2. Translate them (if asked).
    3. Create an Anki deck with them.

    IMPORTANT: When creating the Anki deck, you MUST use the `create_anki_stack` tool.
    Pass the ENTIRE list of translated words to `create_anki_stack` at once.
    Do NOT create cards one by one.
    """)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state.get("source_language"),
        "number_of_words": state.get("number_of_words"),
        "word_difficulty": state.get("word_difficulty"),
        "target_language": state.get("target_language")
    }


async def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))  # Use our clean tools list
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


async def main():
    print("Starting Agent (Batch Mode)...")
    react_graph = await build_graph()

    # Request 5 words to test (safe for Free Tier)
    user_prompt = "Get 10 difficult words in Spanish, translate them to English, and create a Spanish::Easy Anki deck."

    messages = [HumanMessage(content=user_prompt)]
    result = await react_graph.ainvoke({"messages": messages})

    print("-" * 50)
    print(f"Final Output: {result['messages'][-1].content}")
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())