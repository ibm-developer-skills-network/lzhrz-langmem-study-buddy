"""
Step 5: Background Memory.

Refactor: remove the in-turn memory tool calls (hot path) and instead extract
and consolidate memories *after* the conversation session ends.

Hot path (step 3–4):  model calls memory tools → higher per-turn latency
Background (this step): memory runs after session → zero added turn latency

The key change:
  • Remove manage_memory / search_memory tools from the model
  • At turn start, manually search the store and inject results into the prompt
  • At session end ('new' or 'quit'), call create_memory_store_manager on the full history

New concept: create_memory_store_manager

Run: python 05_background_memory.py
"""
from typing import Optional
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from config.settings import settings
from langmem import create_memory_manager, create_memory_store_manager
from 02_profile import UserProfile, build_system_prompt


MODEL = ModelInference(
    model_id="openai/gpt-5-nano",
    credentials=credentials,
    project_id="skills-network",
    params={"temperature": 0, "max_tokens": 200},
)
USER_ID = "student_1"
BASE_SYSTEM_PROMPT = (
    "You are an encouraging study buddy. "
    "Use concrete examples and check understanding with follow-up questions."
)

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

profile_manager = create_memory_manager(
    MODEL, schemas=[UserProfile],
    instructions="Extract student profile info from the conversation.",
    enable_inserts=False,
)


# TODO 2: Create a background_memory_manager using create_memory_store_manager.
#   - model = MODEL
#   - namespace = ("memories", USER_ID)
#   - store = store
#   This manager will automatically search existing memories, extract new ones,
#   and update the store when invoked with a full conversation history.
background_memory_manager = create_memory_store_manager(
    MODEL,
    namespace=("memories", USER_ID),
    store=store,
)


def retrieve_memories(query: str) -> str:
    """Search the store and format memories to inject into the prompt."""
    # TODO 3: Search the store for memories relevant to `query`.
    #   Hint: namespace = ("memories", USER_ID), limit = 5
    #   Format as a <relevant_memories> block.
    results = store.search(("memories", USER_ID), query=query, limit=5)
    if not results:
        return ""
    lines = "\n".join(f"  • {r.value}" for r in results)
    return f"\n\n<relevant_memories>\n{lines}\n</relevant_memories>"


def flush_session(session_history: list) -> None:
    """Run background memory extraction on the completed session."""
    if not session_history:
        return
    print("\n(running background memory extraction...)")
    # TODO 4: Invoke background_memory_manager with the full session history.
    background_memory_manager.invoke({"messages": session_history})
    print("(done)\n")


def chat() -> None:
    model = init_chat_model(MODEL)   # no tools bound — clean model
    history: list[dict] = []
    session_history: list[dict] = []  # accumulates across turns for background flush
    profile_memories: list = []

    print("Study Buddy + Background Memory  —  commands: 'memory', 'new', 'quit'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            flush_session(session_history)
            break
        if user_input.lower() == "new":
            flush_session(session_history)
            session_history.clear()
            history.clear()
            print("--- New conversation ---\n")
            continue
        if user_input.lower() == "memory":
            items = store.search(("memories", USER_ID), query="", limit=20)
            print(f"Store ({len(items)} items): {[i.value for i in items]}\n")
            continue

        history.append({"role": "user", "content": user_input})
        session_history.append({"role": "user", "content": user_input})

        # Build system prompt: profile + memories retrieved for this query
        system_prompt = BASE_SYSTEM_PROMPT
        if profile_memories:
            p = profile_memories[0].value
            lines = "\n".join(f"  {k}: {v}" for k, v in p.items() if v)
            system_prompt += f"\n\n<student_profile>\n{lines}\n</student_profile>"

        # TODO 5: Retrieve relevant memories from the store and add to system_prompt.
        system_prompt += retrieve_memories(user_input)

        response = model.invoke([{"role": "system", "content": system_prompt}, *history])
        reply = response.content
        history.append({"role": "assistant", "content": reply})
        session_history.append({"role": "assistant", "content": reply})
        print(f"\nBuddy: {reply}\n")

        # Profile still updated per-turn (fast, single doc)
        profile_memories = profile_manager.invoke({
            "messages": history[-2:],
            "existing": profile_memories,
        })


if __name__ == "__main__":
    chat()
