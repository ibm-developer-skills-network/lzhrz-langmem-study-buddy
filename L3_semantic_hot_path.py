"""
Step 3: Semantic Memory — Hot Path.

The buddy now has two memory tools it can call *during* each conversation turn:
  • search_memory  — recall relevant facts before answering
  • manage_memory  — save new facts, preferences, or concepts as they come up

"Hot path" = memory operations happen synchronously inside the turn, driven by the LLM.

New concepts: create_manage_memory_tool, create_search_memory_tool, InMemoryStore,
              tool-calling loop

Run: python L3_semantic_hot_path.py
"""
from config.settings import settings
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager
from langgraph.store.memory import InMemoryStore
from L2_profile import UserProfile, build_system_prompt

MODEL = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    api_key=settings.OPENAI_API_KEY,
    max_tokens=200,
    reasoning_effort="minimal"
)

USER_ID = "student_1"
BASE_SYSTEM_PROMPT = (
    "You are an encouraging study buddy. "
    "Before answering, use search_memory to recall relevant facts about this student. "
    "After responding, use manage_memory to save any new facts, concepts, or preferences. "
    "Use concrete examples and occasionally ask follow-up questions."
)

# Create an InMemoryStore with a vector index for semantic search.
store = InMemoryStore(
    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
)


# Create manage_memory and search_memory tools.
manage_memory_tool = create_manage_memory_tool(namespace=("memories", USER_ID), store=store)
search_memory_tool = create_search_memory_tool(namespace=("memories", USER_ID), store=store)
MEMORY_TOOLS = [manage_memory_tool, search_memory_tool]
tools_by_name = {t.name: t for t in MEMORY_TOOLS}


def run_turn(model_with_tools, messages: list) -> tuple[str, list]:
    """Execute one turn, handling any tool calls the model makes."""
    # TODO 2: Implement the tool-calling loop.
    #   1. Invoke model_with_tools with the current messages list
    #   2. Append the response to messages
    #   3. If the response has tool_calls, execute each tool and append ToolMessage results
    #   4. Loop until there are no more tool calls; then return (reply_text, updated_messages)
    #
    #   Hint: ToolMessage(content=str(result), tool_call_id=tc["id"])

    # Fallback (no tools wired up yet): plain invoke
    response = model_with_tools.invoke(messages)
    return response.content, [*messages, response]


def chat() -> None:
    history: list[dict] = []
    profile_memories: list = []

    profile_manager = # TODO 3: Create the profile manager (from step 2) 

    model_with_tools = # TODO 4: Bind memory tools to the model
    # Hint: use MODEL.bind_tools(MEMORY_TOOLS) to create a new model_with_tools.

    # TODO 5: Add a new command 'memory' to inspect the current contents of the store.
    # Hint: print("Study Buddy + Semantic Memory (hot path)  —  commands: ...")

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
        # TODO 5 (continued): Add a new command 'memory' to inspect the current contents of the store.
        # Hint: Use store.search(("memories", USER_ID), query="", limit=20) to get the current items.

        history.append({"role": "user", "content": user_input})

        system_prompt = # TODO 6: Build system prompt using the function we defined in the previous section

        # TODO 7: Run the turn with the model_with_tools and run_turn.
        # Hint: reply, messages = ...

        history.append({"role": "assistant", "content": reply})
        print(f"\nBuddy: {reply}\n")

        profile_memories = profile_manager.invoke({
            "messages": history[-2:],
            "existing": profile_memories,
        })


if __name__ == "__main__":
    chat()
