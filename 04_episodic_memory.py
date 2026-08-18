"""
Step 4: Episodic Memory.

Beyond raw facts, the buddy now captures *how* it successfully explained things —
storing full reasoning chains as reusable examples (few-shot episodes).

Episodic memory is different from semantic memory:
  • Semantic memory  = facts/preferences ("student struggles with recursion")
  • Episodic memory  = past experiences ("last time I used a stack analogy, it clicked")

Episodes accumulate (enable_inserts=True) and are injected as examples into the
system prompt when a similar topic comes up.

New concepts: Episode schema, per-turn episode extraction, episode retrieval

Run: python 04_episodic_memory.py
"""
from typing import Optional
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from config.settings import settings
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from 02_profile import UserProfile
from 03_semantic_hot_path import run_turn

from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager

MODEL = ModelInference(
    model_id="openai/gpt-5-nano",
    credentials=credentials,
    project_id="skills-network",
    params={"temperature": 0, "max_tokens": 200},
)
USER_ID = "student_1"
BASE_SYSTEM_PROMPT = (
    "You are an encouraging study buddy. "
    "Before answering, use search_memory to recall relevant facts about this student. "
    "After responding, use manage_memory to save any new facts or concepts. "
    "Use concrete examples and check understanding with follow-up questions."
)

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

manage_memory_tool = create_manage_memory_tool(namespace=("memories", USER_ID), store=store)
search_memory_tool = create_search_memory_tool(namespace=("memories", USER_ID), store=store)
MEMORY_TOOLS = [manage_memory_tool, search_memory_tool]
tools_by_name = {t.name: t for t in MEMORY_TOOLS}

profile_manager = create_memory_manager(
    MODEL, schemas=[UserProfile],
    instructions="Extract student profile info from the conversation.",
    enable_inserts=False,
)


# TODO 8: Define an Episode Pydantic model that captures a full reasoning chain.
# Suggested fields: topic, observation (what happened / student's question),
# thoughts (buddy's internal reasoning), action (how it explained), result (outcome/reaction)
class Episode(BaseModel):
    """A record of one successful explanation interaction."""
# ── SOLUTION ──────────────────────────────────────────────────────────────────
#     topic: str = Field(..., description="Subject or concept being explained")
#     observation: str = Field(..., description="Context — what the student asked or struggled with")
#     thoughts: str = Field(..., description="Reasoning behind the chosen explanation approach")
#     action: str = Field(..., description="How the buddy explained it (approach, analogy, example used)")
#     result: str = Field(..., description="Student reaction and whether understanding improved")
# ──────────────────────────────────────────────────────────────────────────────


episode_manager = # TODO 9: Create an episode_manager using create_memory_manager.
# ── SOLUTION ──────────────────────────────────────────────────────────────────
# episode_manager = create_memory_manager(
#     MODEL,
#     schemas=[Episode],
#     instructions=(
#         "If the conversation contains a clear explanation exchange, extract it as an episode. "
#         "Focus on what made the explanation effective. Only extract when there's genuine signal."
#     ),
#     enable_inserts=True,
# )
# ──────────────────────────────────────────────────────────────────────────────


def retrieve_episodes(topic: str) -> str:
    """Search the episode store and format relevant past examples."""
    # TODO 10: Search the store (hint: use `store.search`) for past episodes related to `topic`.
    # Format the results as a <past_examples> block to inject into the prompt.
    # namespace = ("episodes", USER_ID), query = topic
    # ── SOLUTION ────────────────────────────────────────────────────────────────
    # results = store.search(("episodes", USER_ID), query=topic, limit=2)
    # if not results:
    #     return ""
    # examples = []
    # for r in results:
    #     e = r.value
    #     examples.append(
    #         f"Topic: {e.get('topic')}\n"
    #         f"Situation: {e.get('observation')}\n"
    #         f"What worked: {e.get('action')}\n"
    #         f"Outcome: {e.get('result')}"
    #     )
    # return "\n\n<past_examples>\n" + "\n---\n".join(examples) + "\n</past_examples>"
    # ────────────────────────────────────────────────────────────────────────────


def chat() -> None:
    model_with_tools = init_chat_model(MODEL).bind_tools(MEMORY_TOOLS)
    history: list[dict] = []
    profile_memories: list = []

    print("Study Buddy + Episodic Memory  —  commands: 'profile', 'memory','new', 'quit'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "new":
            history.clear()
            print("--- New conversation ---\n")
            continue
        if user_input.lower() == "profile":
            print(f"Profile: {[m.value for m in profile_memories] or '(empty)'}\n")
            continue
        if user_input.lower() == "memory":
            facts = store.search(("memories", USER_ID), query="", limit=20)
            episodes = store.search(("episodes", USER_ID), query="", limit=20)
            print(f"Hot Path ({len(facts)}): {[f.value for f in facts]}")
            print(f"Episodes ({len(episodes)}): {[e.value.get('topic') for e in episodes]}\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Build system prompt — profile + relevant episodes (copied from build_system_prompt)
        system_prompt = BASE_SYSTEM_PROMPT
        if profile_memories:
            p = profile_memories[0].value
            lines = "\n".join(f"  {k}: {v}" for k, v in p.items() if v)
            system_prompt += f"\n\n<student_profile>\n{lines}\n</student_profile>"

            # TODO 11: Retrieve relevant past episodes and append to system_prompt.
            # ── SOLUTION ────────────────────────────────────────────────────────────
            # system_prompt += retrieve_episodes(user_input)
            # ────────────────────────────────────────────────────────────────────────

        messages = [{"role": "system", "content": system_prompt}, *history]
        reply, messages = run_turn(model_with_tools, messages)
        history.append({"role": "assistant", "content": reply})
        print(f"\nBuddy: {reply}\n")

        # Update profile
        profile_memories = profile_manager.invoke({
            "messages": history[-2:],
            "existing": profile_memories,
        })

        # TODO 12: Extract a new episode from this turn and store it.
        # Call episode_manager.invoke with messages=history[-2:].
        # Then put each extracted episode into the store under ("episodes", USER_ID).
        # Hint: store.put(namespace, key, value) — use a unique key per episode.
        # ── SOLUTION ────────────────────────────────────────────────────────────
        # new_episodes = episode_manager.invoke({"messages": history[-2:]})
        # for i, ep in enumerate(new_episodes):
        #     key = f"ep_{len(history)}_{i}"
        #     store.put(("episodes", USER_ID), key, ep.value)
        # ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    chat()
