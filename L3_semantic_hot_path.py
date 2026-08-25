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
