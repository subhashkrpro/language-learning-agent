# Language Learning Agent

An intelligent language learning assistant powered by LangChain, LangGraph, and LLMs. This agent helps you learn new languages by generating random words, translating them, and automatically creating Anki decks for vocabulary practice.

## 📋 Overview

The Language Learning Agent is an AI-powered tool that streamlines the vocabulary learning process. It uses multi-agent architecture with tool-based interactions to:

- **Select vocabulary**: Randomly pick words from a curated database of 20+ languages
- **Filter by difficulty**: Choose beginner, intermediate, or advanced words
- **Translate automatically**: Leverage LLMs to translate words between languages
- **Create study materials**: Generate Anki decks for spaced repetition learning

### Anki Integration

- **Direct deck creation**: Automatically create and populate Anki decks
- **Requires AnkiConnect**: Works with Anki running AnkiConnect addon
- **Batch card creation**: Efficiently add multiple cards at once

## 🏗️ Project Structure

```
language-learning-agent/
├── main.py                          # Agent entry point with LangGraph implementation
├── pyproject.toml                   # Python project configuration and dependencies
├── README.md                        # This file
├── agent/
│   └── tools.py                     # Tool definitions for the AI agent
├── data/                            # Cleaned word lists (JSON format)
│   ├── Catalan/
│   ├── Chinese/
│   ├── English/                     # Example: 150 curated English words
│   ├── Spanish/
│   └── ... (20+ languages)
├── raw-word-list/                   # Original unprocessed word data
│   ├── Catalan/
│   ├── Chinese/
│   └── ... (parallel to data/)
└── Notebooks/
    └── clean-word-list.ipynb        # Data cleaning pipeline notebook
```


## 🔧 Core Components

### 1. Main Agent (`main.py`)

The entry point using LangGraph to orchestrate the AI workflow:

- **AgentState**: Tracks conversation state and language learning parameters
- **Tools**: Integrated with the agent for word selection, translation, and Anki operations
- **System Prompt**: Guides the LLM to complete language learning tasks efficiently

```python
# Example usage
user_prompt = "Get 10 intermediate words in korean, translate them to Spanish, and create a Spanish::Easy Anki deck."
```

**Features:**

- Async/await support for non-blocking operations
- Conditional edge routing (tools_condition)
- Clean separation between agent logic and tool definitions

### 2. Tools (`agent/tools.py`)

Four main tools available to the agent:

#### `get_n_random_words(language, n)`

Retrieves n random words from a language's word list.

- **Parameters**:
  - `language`: Language name (e.g., "English", "Spanish")
  - `n`: Number of words to retrieve
- **Returns**: List of random words

#### `get_n_random_words_by_difficulty_level(language, difficulty_level, n)`

Retrieves random words filtered by difficulty.

- **Parameters**:
  - `language`: Target language
  - `difficulty_level`: "beginner", "intermediate", or "advanced"
  - `n`: Number of words
- **Returns**: List of words at specified difficulty

#### `translate_words(random_words, source_language, target_language)`

Translates words between languages using an LLM.

- **Parameters**:
  - `random_words`: List of words to translate
  - `source_language`: Original language
  - `target_language`: Destination language
- **Returns**: Dictionary with translations structured as:
  ```json
  {
    "translations": [
      {"source": "word", "target": "translation"},
      ...
    ]
  }
  ```

#### `create_anki_stack(deck_name, cards)`

Creates an Anki deck and adds cards in batch.

- **Parameters**:
  - `deck_name`: Name of the deck (e.g., "Spanish::Easy")
  - `cards`: List of card dictionaries with "source" and "target" fields
- **Returns**: Success message with card count and error information
- **Requirements**: Anki running with AnkiConnect addon on localhost:8765

### 3. LLM Configuration

The agent supports multiple LLM providers configured via environment variables:

```bash
# In .env file
LLM_PROVIDER=gemini        # Options: gemini, openai, ollama
LLM_MODEL=gemini-2.5-flash # Model specification (optional, uses defaults if not set)
```

**Supported Providers:**

- **Gemini**: `ChatGoogleGenerativeAI` (default model: gemini-2.5-flash)
- **OpenAI**: `ChatOpenAI` (default model: gpt-4o)
- **Ollama**: `ChatOllama` (default model: llama3.2:3b, requires local Ollama)

**Factory Function**: `get_translation_model()` abstracts provider selection

## 📊 Data Pipeline

### Word List Cleaning (`Notebooks/clean-word-list.ipynb`)

The Jupyter notebook processes raw word lists through:

1. **Loading**: Import raw word data from `raw-word-list/` directory
2. **Cleaning**: Remove duplicates, normalize encoding, filter valid words
3. **Categorization**: Assign difficulty levels (beginner/intermediate/advanced) based on frequency analysis
4. **Export**: Save processed data to `data/` as JSON files

**Notebook Cells**: 22 cells covering:

- Library imports and setup
- Data loading and exploration
- Cleaning operations
- Difficulty level assignment
- Quality checks
- Export to final format

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- uv package manager
- Anki with AnkiConnect addon for deck creation
- Anki Desktop Application

### Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd language-learning-agent
   ```
2. **Create virtual environment** (recommended)

   ```bash
   uv sync
   ```
3. **Set up environment variables**

   ```
   Change .env.example to .env and inside your .env file add your provider and API Key
    
     GOOGLE_API_KEY=your_api_key
     LLM_PROVIDER=gemini_or_openai

   ```

4. **Start Anki desktop application and make sure anki addon also installed**

5. **Run Agent**

   ```bash
    uv run python main.py
   ```
**Anki Integration:**

If you want to create Anki decks:

1. Install [Anki](https://apps.ankiweb.net/)
2. Install [AnkiConnect addon](https://github.com/FooSoft/anki-connect)
3. Restart Anki (it runs on localhost:8765 by default)

**Example Output:**

```
Starting Agent (Batch Mode)...
--------------------------------------------------
Final Output: Successfully created Anki deck "Spanish::Intermediate" 
with 10 Korean→Spanish translation cards.
--------------------------------------------------
```

## 🛠️ Dependencies

Key dependencies (see `pyproject.toml` for full list):

| Package                    | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| `langchain-core`         | Core LLM abstraction and schema               |
| `langgraph`              | Multi-agent orchestration and routing         |
| `langchain-google-genai` | Google Gemini integration                     |
| `langchain-openai`       | OpenAI GPT integration                        |
| `langchain-ollama`       | Ollama local LLM support                      |
| `spacy`                  | NLP utilities                                 |
| `python-dotenv`          | Environment variable management               |
| `requests`               | HTTP client for Anki integration              |
| `wordfreq`               | Word frequency analysis for difficulty levels |
| `pandas`                 | Data manipulation                             |
| `jupyter`                | Notebook environment                          |

## 📈 Workflow Example

### Scenario: Create a Spanish study deck from Korean words

```
User Input:
"Get 10 intermediate words in korean, translate them to Spanish, 
and create a Spanish::Easy Anki deck."

Agent Workflow:
1. Call get_n_random_words_by_difficulty_level("Korean", "intermediate", 10)
   → Returns: ["한국", "의미", "언어", ...]

2. Call translate_words(korean_words, "Korean", "Spanish")
   → Returns: {"translations": [{"source": "한국", "target": "Corea"}, ...]}

3. Call create_anki_stack("Spanish::Easy", cards)
   → Returns: "Successfully created deck with 10 cards"

4. Agent reports: Deck created and ready for study!
```

## 🐛 Troubleshooting

### "Error: File not found at data/..."

- Ensure the language directory exists in `data/`
- Verify file spelling (case-sensitive)

### "Error connecting to Anki"

- Is Anki running? (Should be visible in taskbar)
- Is AnkiConnect addon installed?
- Check Anki is listening on http://127.0.0.1:8765

### "Invalid JSON response from LLM"

- Some LLMs may have formatting issues
- Try a different model or LLM provider
- Check that the prompt is clear

### "No words found for [difficulty] in [language]"

- Not all languages have words for all difficulty levels
- Try "beginner" or check available words with a simpler request


## 🤝 Contributing

Contributions welcome! Areas of interest:

- Adding new language word lists
- Improving the data cleaning pipeline
- Enhancing translation quality
- Adding new LLM providers
- Bug fixes and optimizations

## 📚 References

- [YouTube](https://youtu.be/j4sNAwrx3kc)
- [Repository: all-words-in-all-languages](https://github.com/eymenefealtun/all-words-in-all-languages)
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anki Documentation](https://docs.ankiweb.net/)
- [AnkiConnect GitHub](https://github.com/FooSoft/anki-connect)
- [Spacy Documentation](https://spacy.io/)
