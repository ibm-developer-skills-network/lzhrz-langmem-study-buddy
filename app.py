"""
LangMem Study Buddy — Web UI
Run: python app.py
Open: http://localhost:5000

Mirrors the final chatbot from L5_prompt_optimization.py: profile memory,
hot-path search_memory/manage_memory tools, episodic memory, and
self-reflective prompt optimization.
"""
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from config.settings import settings
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from L2_profile import UserProfile
from L3_semantic_hot_path import run_turn
from L4_episodic_memory import Episode, retrieve_episodes

from langmem import create_memory_manager, create_manage_memory_tool, create_search_memory_tool, create_prompt_optimizer

MODEL = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
    max_tokens=200,
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

    # Self-reflect on the session and run the prompt optimizer.
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

app = Flask(__name__)

g = {
    "history": [],
    "session_history": [],
    "profile_memories": [],
    "system_prompt": None,
}
c = {}  # components


def init() -> None:
    c["model"] = MODEL
    c["model_with_tools"] = c["model"].bind_tools(MEMORY_TOOLS)
    g["system_prompt"] = load_system_prompt()

def snapshot() -> dict:
    return {
        "profile": g["profile_memories"][0].value if g["profile_memories"] else {},
        "facts": [i.value for i in store.search(("memories", USER_ID), query="", limit=30)],
        "episodes": [
            {"topic": e.value.get("topic", "?"), "action": e.value.get("action", "")[:120]}
            for e in store.search(("episodes", USER_ID), query="", limit=20)
        ],
        "system_prompt": g["system_prompt"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({"memory": snapshot()})


@app.route("/chat", methods=["POST"])
def chat_route():
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    g["history"].append({"role": "user", "content": msg})
    g["session_history"].append({"role": "user", "content": msg})

    full_prompt = g["system_prompt"]
    if g["profile_memories"]:
        p = g["profile_memories"][0].value
        lines = "\n".join(f"  {k}: {v}" for k, v in p.items() if v)
        full_prompt += f"\n\n<student_profile>\n{lines}\n</student_profile>"
        full_prompt += retrieve_episodes(msg)

    messages = [{"role": "system", "content": full_prompt}, *g["history"]]
    reply, _ = run_turn(c["model_with_tools"], messages)
    g["history"].append({"role": "assistant", "content": reply})
    g["session_history"].append({"role": "assistant", "content": reply})

    g["profile_memories"] = profile_manager.invoke({
        "messages": g["history"][-2:],
        "existing": g["profile_memories"],
    })

    new_episodes = episode_manager.invoke({"messages": g["history"][-2:]})
    for i, ep in enumerate(new_episodes):
        key = f"ep_{len(g['history'])}_{i}"
        store.put(("episodes", USER_ID), key, ep.value)

    return jsonify({"reply": reply, "memory": snapshot()})


@app.route("/reset", methods=["POST"])
def reset():
    # Flush the session — self-reflect and optimize the system prompt before clearing.
    g["system_prompt"] = flush_session(c["model"], g["session_history"], g["system_prompt"])
    g["history"].clear()
    g["session_history"].clear()
    return jsonify({"ok": True, "memory": snapshot()})


if __name__ == "__main__":
    print("Starting LangMem Study Buddy...")
    init()
    print("Open http://localhost:5000")
    app.run(port=5000, debug=False)
