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
