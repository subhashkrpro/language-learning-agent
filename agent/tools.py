import os
import json
import random
import re
import requests
from typing import List, Dict, Any, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import supported LLMs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

# --- Configuration ---
ANKI_CONNECT_URL = "http://127.0.0.1:8765"


def get_translation_model():
    """Factory to get the translation model based on environment variables."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model_name = os.getenv("LLM_MODEL")

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-flash",
            temperature=0.3
        )
    elif provider == "openai":
        return ChatOpenAI(model=model_name or "gpt-4o", temperature=0.3)
    elif provider == "ollama":
        return ChatOllama(model=model_name or "llama3.2:3b", temperature=0.3)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


# --- TOOLS ---

@tool
def get_n_random_words(language: str, n: int) -> List[str]:
    """Selects a specified number of random words from a language-specific word list."""
    n = int(n)
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    if not os.path.exists(path):
        return [f"Error: File not found at {path}"]

    with open(path, 'r', encoding='utf-8') as f:
        word_list = json.load(f)

    keys = list(word_list.keys())
    if n > len(keys): n = len(keys)

    random_keys = random.sample(keys, n)
    return [word_list[k]["word"] for k in random_keys]


@tool
def get_n_random_words_by_difficulty_level(language: str, difficulty_level: str, n: int) -> List[str]:
    """Retrieves random words filtered by difficulty (beginner, intermediate, advanced)."""
    n = int(n)
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    if not os.path.exists(path):
        return [f"Error: File not found at {path}"]

    with open(path, 'r', encoding='utf-8') as f:
        word_list = json.load(f)

    filtered = {k: v for k, v in word_list.items() if v.get("word_difficulty", "").lower() == difficulty_level.lower()}

    if not filtered:
        return [f"Error: No words found for {difficulty_level} in {language}"]

    keys = list(filtered.keys())
    if n > len(keys): n = len(keys)

    random_keys = random.sample(keys, n)
    return [filtered[k]["word"] for k in random_keys]


@tool
def translate_words(random_words: List[str], source_language: str, target_language: str) -> Dict[str, Any]:
    """Translates a list of words. Returns a dictionary with 'translations' list."""
    print(f"DEBUG: Translating {len(random_words)} words...")
    model = get_translation_model()

    prompt = (
        f"Translate these {len(random_words)} words from {source_language} to {target_language}.\n"
        f"Return ONLY valid JSON: {{ \"translations\": [ {{ \"source\": \"word\", \"target\": \"translation\" }} ] }}\n"
        f"Words: {json.dumps(random_words, ensure_ascii=False)}"
    )

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        text = response.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "Invalid JSON response", "raw": text}
    except Exception as e:
        return {"error": str(e)}


@tool
def create_anki_stack(deck_name: str, cards: List[Dict[str, str]]) -> str:
    """
    Creates an Anki deck and adds multiple cards in ONE go.

    Args:
        deck_name: Name of the deck (e.g., "Spanish::Easy")
        cards: List of dicts, e.g., [{"front": "hola", "back": "hello"}, ...]
    """
    print(f"DEBUG: Batch creating deck '{deck_name}' with {len(cards)} cards...")

    # 1. Create Deck
    try:
        requests.post(ANKI_CONNECT_URL, json={
            "action": "createDeck",
            "version": 6,
            "params": {"deck": deck_name}
        })
    except Exception as e:
        return f"Error connecting to Anki: {e}. Is Anki running?"

    # 2. Add Cards (Local loop, super fast)
    success_count = 0
    errors = []

    for card in cards:
        payload = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Front": card.get("source", card.get("front", "")),
                        "Back": card.get("target", card.get("back", ""))
                    }
                }
            }
        }
        resp = requests.post(ANKI_CONNECT_URL, json=payload).json()
        if resp.get("error"):
            errors.append(resp["error"])
        else:
            success_count += 1

    return f"Success! Created deck '{deck_name}' and added {success_count} cards. Errors: {len(errors)}"