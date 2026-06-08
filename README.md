# LangGraph Agentic RAG System

An autonomous multi-agent system that plans what to retrieve, retrieves it, reasons over the result, and responds — without human steps in between.

## What makes this agentic (not just RAG)

Basic RAG: question → retrieve → answer (one fixed path)

This system: question → agent decides retrieval strategy → retrieves → evaluates quality → re-retrieves if needed → synthesises → answers

The agent can use web search OR document retrieval depending on what the question needs. It chooses. That's the difference.

## Architecture

```
User question
     ↓
 LangGraph agent (state machine)
     ↓
 Route: web search OR document retrieval
     ↓
 Retrieve chunks / search results
     ↓
 Evaluate relevance
     ↓
 Re-route if low quality (self-correcting loop)
     ↓
 Synthesise and respond
```

## Files

| File | What it does |
|------|-------------|
| `agentic_rag.ipynb` | Main agentic RAG system — start here |
| `rag_agent.py` | Python script version of the RAG agent |
| `qna_chatbot_using_langgraph.py` | QnA chatbot with memory using LangGraph |
| `google_search_engine.py` | Web search tool integration |
| `basic_langgraph.ipynb` | LangGraph fundamentals (nodes, edges, state) |
| `basic_agents.ipynb` | Agent building basics |

## Tech stack

- LangGraph (agent orchestration, state machine)
- LangChain (LLM chains, document loaders)
- OpenAI GPT-4o (reasoning)
- Groq LLaMA (fast inference)
- Python

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Run `agentic_rag.ipynb` in Jupyter or VS Code.

## Related projects

- [RAG PDF Chatbot](https://github.com/Saiajaykumar12/langchain-rag-pdf-chatbot) — basic RAG without the agent layer
- [SQL AI Chatbot](https://github.com/Saiajaykumar12/sql-ai-chatbot-langchain) — natural language to SQL
