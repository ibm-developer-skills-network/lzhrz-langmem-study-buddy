"""
Step 1: Basic Study Buddy — no memory.

The agent answers questions but forgets everything when restarted.
This is the baseline we'll improve in every subsequent step.

Run: python L1_basic_chat.py
"""
from config.settings import settings
from langchain.chat_models import init_chat_model

MODEL = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    api_key=settings.OPENAI_API_KEY,
    max_tokens=200,
    reasoning_effort="minimal"
)

SYSTEM_PROMPT = (
    "You are an encouraging study buddy. Help the student understand topics clearly, "
    "use concrete examples, and occasionally ask follow-up questions to check understanding."
)


def chat() -> None:
    history: list[dict] = []

    print("Study Buddy  —  commands: 'new' = fresh conversation, 'quit' = exit\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            history.clear()
            print("--- New conversation ---\n")
            continue

        history.append({"role": "user", "content": user_input})
        response = MODEL.invoke([{"role": "system", "content": SYSTEM_PROMPT}, *history])
        reply = response.content
        history.append({"role": "assistant", "content": reply})
        print(f"\nBuddy: {reply}\n")


if __name__ == "__main__":
    chat()
