"""
Step 2: User Profile Memory.

The buddy maintains a single profile document for the student and personalizes
responses with it. Profiles update in-place (enable_inserts=False) rather than
growing a list of separate entries.

Key concept: create_memory_manager with enable_inserts=False

Run: python 02_profile.py
"""
from typing import Optional
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from config.settings import settings
from langchain.chat_models import init_chat_model
from langmem import create_memory_manager
from pydantic import BaseModel, Field

credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com")
client = APIClient(credentials)

MODEL = ModelInference(
    model_id="openai/gpt-5-nano",
    credentials=credentials,
    project_id="skills-network",
    params={"temperature": 0, "max_tokens": 200},
)

BASE_SYSTEM_PROMPT = (
    "You are an encouraging study buddy. Help the student understand topics clearly, "
    "use concrete examples, and occasionally ask follow-up questions to check understanding."
)

# Define a UserProfile Pydantic model.
# Suggested fields: name, grade_level, subjects, learning_style, known_struggles, goals
class UserProfile(BaseModel):
    """Everything we know about the student."""
    name: Optional[str] = None
    grade_level: Optional[str] = None
    subjects: list[str] = Field(default_factory=list, description="Subjects currently being studied")
    learning_style: Optional[str] = Field(None, description="e.g. visual, example-first, step-by-step")
    known_struggles: list[str] = Field(default_factory=list, description="Topics the student finds hard")
    goals: Optional[str] = None


# TODO 1: Create a profile_manager using create_memory_manager.
#   - model = MODEL
#   - schemas = [UserProfile]
#   - enable_inserts = False   ← one profile doc, always updated in place
#   - Write instructions to extract student info from the conversation
profile_manager = create_memory_manager(
    MODEL,
    schemas=[UserProfile],
    instructions=(
        "Extract information about the student from the conversation. "
        "Update the profile with any new details; do not duplicate existing info."
    ),
    enable_inserts=False,
)


def build_system_prompt(profile_memories: list) -> str:
    # If we have a profile, append a <student_profile> block to the prompt.
    if profile_memories:
        profile = profile_memories[0].value          # single document
        fields = "\n".join(f"  {k}: {v}" for k, v in profile.items() if v)
        return BASE_SYSTEM_PROMPT + f"\n\n<student_profile>\n{fields}\n</student_profile>"
    return BASE_SYSTEM_PROMPT


def chat() -> None:
    model = init_chat_model(MODEL)
    history: list[dict] = []
    profile_memories: list = []      # holds the single profile memory entry

    print("Study Buddy + Profile  —  commands: 'profile', 'new', 'quit'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "new":
            history.clear()
            print("--- New conversation (profile persists) ---\n")
            continue
        if user_input.lower() == "profile":
            print(f"Profile: {[m.value for m in profile_memories] or '(empty)'}\n")
            continue

        history.append({"role": "user", "content": user_input})
        response = model.invoke([
            {"role": "system", "content": build_system_prompt(profile_memories)},
            *history,
        ])
        reply = response.content
        history.append({"role": "assistant", "content": reply})
        print(f"\nBuddy: {reply}\n")

        # After each turn, call profile_manager to update the profile.
        # Pass messages=history[-2:] (just this turn) and existing=profile_memories.
        # Store the returned list back into profile_memories.
        profile_memories = profile_manager.invoke({
            "messages": history[-2:],
            "existing": profile_memories,
        })


if __name__ == "__main__":
    chat()
