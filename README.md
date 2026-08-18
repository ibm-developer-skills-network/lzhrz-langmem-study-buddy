# LangMem Study Buddy

A step-by-step tutorial for learning [LangMem](https://langchain-ai.github.io/langmem/) by building a study buddy agent.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
```

All steps need `OPENAI_API_KEY` (for the LLM). Steps 3–6 also use it for vector embeddings in `InMemoryStore`.

## Steps

| File | Feature |
|------|---------|
| `L1_basic_chat.py` | Plain chat loop — no memory |
| `L2_profile.py` | User profile with `create_memory_manager` |
| `L3_semantic_hot_path.py` | Semantic memory via in-turn tool calls |
| `L4_episodic_memory.py` | Episodic memory from past explanations |
| `L5_prompt_optimization.py` | Autonomous prompt optimization |

## How to use each file

Each file is runnable as-is (the starter code works without the memory features).  
Every `TODO` block has the solution commented out directly below it — uncomment to check your work.

## Chat commands

All steps support these commands while running:
- `new` — start a fresh conversation (memory persists across conversations)
- `quit` — exit
- `memory` (steps 3+) — print current store contents
