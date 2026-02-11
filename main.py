import asyncio
from typing import TypedDict, Optional, Annotated

from langchain_core.messages import AnyMessage,SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words
)

from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    target_language: Optional[str]

CLANKI_JS = "/Users/Jodie.Burchell/Documents/git/clanki/build/index.js"

local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words
]

async def setup_tools():
    client = MultiServerMCPClient(
        {
            "clanki": {
                "command": "node",
                "args": [CLANKI_JS],
                "transport": "stdio",
            }
        }
    )

    mcp_tools = await client.get_tools()
    return [*local_tools, *mcp_tools]


def assistant(state: AgentState):
    textual_description_of_tools = """
    def get_n_random_words(language: str,
                       n: int, ) -> list:
    Get a specified number of random words from a word list file for a given language.

    This function fetches a language-specific word list from a JSON file and
    selects a random subset of words from this list, returning it as a list.

    Args:
        word_list: A dictionary of words.
        n: The desired number of words.

    Returns:
        list: A word list as a list.

    def get_n_random_words_by_difficulty_level(language: str,
                                           difficulty_level: str,
                                           n: int
                                           ) -> list:
    Retrieves a specified number of random words from a word list filtered by a given
    difficulty level within a specific language. This function expects the word list to
    be stored in a JSON file, organized by language, in the 'data' directory.

    :param language: The language of the word list to search within.
    :type language: str
    :param difficulty_level: The difficulty level of the words. This must be one of `beginner`,
        `intermediate` or `advanced`.
    :type difficulty_level: str
    :param n: The number of random words to retrieve.
    :type n: int
    :return: A list of randomly selected words matching the specified difficulty
        level.
    :rtype: list

    def translate_words(random_words: list, source_language: str, target_language: str) -> dict:
    Translate a list of words from a source language to a target language using a translation
    model. The returned translations are provided in the same order as the input words.

    :param random_words: List of words to be translated.
        The words must be provided as strings in the source language.
    :param source_language: Source language of the words in random_words.
    :param target_language: Target language into which the words should be translated.
    :return: A dictionary containing the translations:
        {
            "translations": [
                {"source": "<original_word>", "target": "<translated_word>"},
                ...
            ]
        }
    """

    sys_msg = SystemMessage(content=f"""
        You are a helpful language learning assistant. You can carry out actions using the following tools: {textual_description_of_tools}. 

        The user is going to give you a command.

        Your job is to check:
        1. Which source language the user wants words from.
        2. How many words they want.
        3. Whether they want words of a specific difficulty, part-of-speech, or just random words.
        4. Whether they want these words translated into a target language.
        5. Whether they want to add these words to an Anki deck. Make sure the `create-deck` tool is called before `create-card`.

        Here are some example workflows:
        input: Get 20 random words in Spanish.
        source language: Spanish
        number of words: 20

        input: Get 10 hard words in German
        source language: German
        number of words: 10
        word difficulty: advanced

        input: Get 20 easy words in Spanish, translate them to English, and create a new Anki deck with them called Spanish::Easy
        source language: Spanish
        target language: English
        number of words: 20
        word difficulty: beginner
        tools workflow: get_n_random_words_by_difficulty_level -> translate_words -> mcp_tools::create_deck -> mcp_tools::create_card

        input: Get 10 random words in German, and create a new Anki deck with them called German::Words
        source language: German
        number of words: 10
        tools workflow: get_n_random_words -> mcp_tools::create_deck -> mcp_tools::create_card
        """)

    # Access tools from the state or a global variable
    tools = assistant.tools if hasattr(assistant, "tools") else []
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.7
    )
    llm_with_tools = llm.bind_tools(tools)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state["source_language"],
        "number_of_words": state["number_of_words"],
        "word_difficulty": state["word_difficulty"],
        "target_language": state["target_language"]
    }

async def build_graph():
    """Build the state graph with properly initialized tools."""
    tools = await setup_tools()
    assistant.tools = tools  # Store tools for access in assistant function

    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    builder.add_edge("tools", "assistant")

    return builder.compile()


async def main():
    """Main async function to run the application."""
    react_graph = await build_graph()

    user_prompt = "Please get 10 easy words in Spanish, translate them to English, and create a new Anki deck with them called Spanish::Easy."

    messages = [HumanMessage(content=user_prompt)]

    # Use ainvoke instead of invoke for async execution
    result = await react_graph.ainvoke({
        "messages": messages,
        "source_language": None,
        "number_of_words": None,
        "word_difficulty": None,
        "target_language": None
    })

    print(f"Final messages: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
