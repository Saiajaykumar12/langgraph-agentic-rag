# LangGraph Agentic RAG System

An autonomous AI agent built with LangGraph that decides *how* to retrieve
information, reasons over it across multiple steps, and responds —
moving beyond fixed RAG pipelines into true agentic behaviour.

## Basic RAG vs Agentic RAG

| | Basic RAG | Agentic RAG (this repo) |
|---|---|---|
| Retrieval | Always retrieves from vector store | Agent decides whether to retrieve, search web, or answer from memory |
| Steps | Fixed: retrieve → generate | Dynamic: agent plans multi-step actions |
| Tools | Single retriever | Multiple tools: retriever + web search + memory |
| Control flow | Linear | Graph-based with conditional edges |

## What's Inside

| File | What It Covers |
|------|---------------|
| `basic_langgraph.ipynb` | LangGraph fundamentals — nodes, edges, state, conditional routing |
| `basic_agents.ipynb` | Building agents with tool calling — how agents decide what action to take |
| `agentic_rag.ipynb` | Full agentic RAG system — agent chooses between retrieval and web search |
| `qna_chatbot_using_langgraph.py` | Stateful QnA chatbot with conversation memory using LangGraph |
| `rag_agent.py` | Script version of the agentic RAG pipeline |
| `google_search_engine.py` | Google search tool integration for real-time web retrieval |

## Agent Architecture
User Question
↓
LangGraph Agent Node
↓
Agent decides action:
├── Retrieve from vector store (for document questions)
├── Search the web (for current/external information)
└── Answer directly (if sufficient context in memory)
↓
Tool execution
↓
Agent evaluates result → loops back if more steps needed
↓
Final Answer
## Key Concepts Demonstrated

- **StateGraph** — how LangGraph manages agent state across multiple steps
- **Conditional edges** — how the agent routes between tools based on reasoning
- **Tool binding** — attaching retriever and web search as callable tools
- **Memory** — maintaining conversation context across turns in the QnA chatbot
- **Agentic loops** — agent re-invokes tools when initial retrieval is insufficient

## Tech Stack

| Tool | Role |
|------|------|
| LangGraph | Agent graph orchestration and state management |
| LangChain | Document loading, splitting, retrieval chains |
| OpenAI GPT | Reasoning and response generation |
| Groq LLaMA | Alternative LLM for faster inference |
| Google Search API | Real-time web retrieval tool |
| FAISS | Vector store for document retrieval |

## Setup

```bash
git clone https://github.com/Saiajaykumar12/langgraph-agentic-rag
cd langgraph-agentic-rag
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

Run notebooks in this order for best understanding:
1. `basic_langgraph.ipynb`
2. `basic_agents.ipynb`
3. `agentic_rag.ipynb`

## Environment Variables
OPENAI_API_KEY=your_openai_key

GROQ_API_KEY=your_groq_key

GOOGLE_API_KEY=your_google_key

GOOGLE_CSE_ID=your_custom_search_engine_id
