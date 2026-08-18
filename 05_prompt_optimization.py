"""
Step 5: Autonomous Prompt Optimization.

The buddy now improves its own system prompt after each session based on how
well the conversations went. This is LangMem's "procedural memory" — the agent
learns *how to behave* rather than just *what to remember*.

The optimizer analyses (conversation, feedback) pairs called "trajectories" and
rewrites the system prompt to bake in what worked. Feedback here comes from the
model's own self-reflection on the session rather than a user-provided rating.

New concept: create_prompt_optimizer, trajectories, self-reflective feedback

Run: python 05_prompt_optimization.py
"""
import json
from pathlib import Path
from typing import Optional
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from config.settings import settings
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from 02_profile import UserProfile
from 03_semantic_hot_path import run_turn
from 04_episodic_memory import Episode, retrieve_episodes

from langmem import create_memory_manager, create_manage_memory_tool, create_search_memory_tool, create_prompt_optimizer

MODEL = ModelInference(
    model_id="openai/gpt-5-nano",
    credentials=credentials,
    project_id="skills-network",
    params={"temperature": 0, "max_tokens": 200},
)
USER_ID = "student_1"
PROMPT_FILE = Path("system_prompt.json")   # persist the optimized prompt across runs

DEFAULT_SYSTEM_PROMPT = (
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

episode_manager = create_memory_manager(
    MODEL,
    schemas=[Episode],
    instructions=(
        "If the conversation contains a clear explanation exchange, extract it as an episode. "
        "Focus on what made the explanation effective. Only extract when there's genuine signal."
    ),
    enable_inserts=True,
)

optimizer = # TODO 13: Create a prompt optimizer using create_prompt_optimizer.
# Hint: create_prompt_optimizer takes parameters 'model' and 'kind'
# 'kind' has options "metaprompt", "gradient", "prompt_memory". Choose "metaprompt".
optimizer = create_prompt_optimizer(MODEL, kind="metaprompt")


def load_system_prompt() -> str:
    if PROMPT_FILE.exists():
        return json.loads(PROMPT_FILE.read_text())["prompt"]
    return DEFAULT_SYSTEM_PROMPT


def save_system_prompt(prompt: str) -> None:
    PROMPT_FILE.write_text(json.dumps({"prompt": prompt}))


def self_reflect(model, session_history: list) -> dict:
    """Ask the model to critique its own performance in the completed session."""
    reflection_prompt = (
        "The conversation above is a tutoring session you (the study buddy) just finished. "
        "Reflect honestly on your own performance: what worked, what didn't land, and what "
        "you should change about your approach next time. Be specific and concise."
    )
    response = model.invoke([*session_history, {"role": "user", "content": reflection_prompt}])
    return {"self_reflection": response.content}


def flush_session(model, session_history: list, system_prompt: str) -> str:
    """Self-reflect on the completed session and run prompt optimization; return updated prompt."""
    if not session_history:
        return system_prompt

    # TODO 14: Self-reflect on the session and run the prompt optimizer.
    #   1. Get feedback via self_reflect(model, session_history)
    #   2. Call optimizer.invoke with trajectories=[(session_history, feedback)]
    #      and prompt=system_prompt
    #   3. The result has an "prompt" key — extract and return it
    #   4. Save the updated prompt with save_system_prompt()
    feedback = self_reflect(model, session_history)
    print(f"(self-reflection: {feedback['self_reflection'][:160]}...)")
    print("(optimizing system prompt...)")
    result = optimizer.invoke({
        "trajectories": [(session_history, feedback)],
        "prompt": system_prompt,
    })
    updated_prompt = result["prompt"]
    save_system_prompt(updated_prompt)
    print(f"(prompt updated)\n")
    return updated_prompt

def chat() -> None:
    model = init_chat_model(MODEL)
    model_with_tools = model.bind_tools(MEMORY_TOOLS)
    # TODO 15: Retrieve the system prompt from persistent storage.
    system_prompt = load_system_prompt()
    history: list[dict] = []
    session_history: list[dict] = [] # new
    profile_memories: list = []

    print("Study Buddy + Prompt Optimization  —  commands: 'prompt', 'profile', 'memory', 'new', 'quit'\n")
    print(f"Current system prompt:\n  {system_prompt[:120]}...\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            # TODO 16: Flush the session and update the system prompt before quitting.
            system_prompt = flush_session(model, session_history, system_prompt)
            break
        if user_input.lower() == "new":
            # TODO 17: Flush the session and update the system prompt & session history before starting a new conversation.
            system_prompt = flush_session(model, session_history, system_prompt)
            session_history.clear()
            history.clear()
            print("--- New conversation ---\n")
            continue
        if user_input.lower() == "profile":
            print(f"\nProfile:\n{profile_memories}\n")
            continue
        # New prompt command to inspect the current system prompt
        if user_input.lower() == "prompt":
            print(f"\nSystem prompt:\n{system_prompt}\n")
            continue
        if user_input.lower() == "memory":
            items = store.search(("memories", USER_ID), query="", limit=20)
            episodes = store.search(("episodes", USER_ID), query="", limit=20)
            print(f"Hot path ({len(items)}): {[i.value for i in items]}")
            print(f"Episodes ({len(episodes)}): {[e.value.get('topic') for e in episodes]}\n")
            continue

        history.append({"role": "user", "content": user_input})
        session_history.append({"role": "user", "content": user_input}) # new

        # Build full system prompt for this turn using the system prompt we loaded
        full_prompt = system_prompt
        if profile_memories:
            p = profile_memories[0].value
            lines = "\n".join(f"  {k}: {v}" for k, v in p.items() if v)
            full_prompt += f"\n\n<student_profile>\n{lines}\n</student_profile>"
            full_prompt += retrieve_episodes(user_input)

        messages = [{"role": "system", "content": full_prompt}, *history]
        reply, _ = run_turn(model_with_tools, messages)
        history.append({"role": "assistant", "content": reply})
        session_history.append({"role": "assistant", "content": reply}) # new
        print(f"\nBuddy: {reply}\n")

        profile_memories = profile_manager.invoke({
            "messages": history[-2:],
            "existing": profile_memories,
        })

        # Extract a new episode from this turn and store it.
        new_episodes = episode_manager.invoke({"messages": history[-2:]})
        for i, ep in enumerate(new_episodes):
            key = f"ep_{len(history)}_{i}"
            store.put(("episodes", USER_ID), key, ep.value)


if __name__ == "__main__":
    chat()
