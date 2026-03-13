"""
Graph RAG — Interactive Chat

Run this script to start a chat session backed by the RAG knowledge base.
The active LLM is set by LLM_PROVIDER in .env:
  - lmstudio  (default) — fully offline, LM Studio must be running
  - claude              — Anthropic API, requires ANTHROPIC_API_KEY

Usage:
    py chat.py
    py chat.py --provider claude
    py chat.py --provider lmstudio

In-session commands:
    /ingest <path>    Add a document to the knowledge base
    /stats            Show database statistics
    /provider         Show or switch active LLM provider
    /clear            Clear conversation history
    /help             Show commands
    quit / exit       Exit
"""

from __future__ import annotations
import sys
import os
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _hr(char="-", width=60):
    print(char * width)


def _print_banner(provider: str):
    _hr("=")
    print("  Graph RAG Chat")
    print(f"  Provider : {provider.upper()}")
    print("  Type your question, or /help for commands.")
    _hr("=")


def _print_help():
    print("""
Commands:
  /ingest <path>   Ingest a file into the knowledge base
  /stats           Show document / chunk / graph counts
  /provider        Show active LLM provider
  /clear           Clear conversation history
  /help            Show this help
  quit / exit      Exit the session
""")


def run_chat(provider_override: str | None = None):
    if provider_override:
        os.environ["LLM_PROVIDER"] = provider_override

    from config import settings
    from rag.pipeline import answer
    from db.connection import get_conn
    from db.queries import get_stats
    from llm.client import active_provider

    provider = active_provider()
    _print_banner(provider)

    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        if user_input == "/help":
            _print_help()
            continue

        if user_input == "/clear":
            history.clear()
            print("History cleared.")
            continue

        if user_input == "/stats":
            try:
                with get_conn() as conn:
                    stats = get_stats(conn)
                for k, v in stats.items():
                    print(f"  {k:<15} {v}")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        if user_input == "/provider":
            print(f"  Active provider: {active_provider().upper()}")
            continue

        if user_input.startswith("/ingest "):
            path = user_input[8:].strip()
            if not path:
                print("  Usage: /ingest <file path>")
                continue
            try:
                from ingestion.pipeline import ingest
                print(f"  Ingesting: {path} ...")
                result = ingest(path)
                print(f"  Done — {result['chunks']} chunks added for '{result['title']}'")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        if user_input.startswith("/"):
            print(f"  Unknown command. Type /help for options.")
            continue

        print()
        try:
            result = answer(user_input)
        except RuntimeError as e:
            print(f"[Error] {e}")
            continue
        except Exception as e:
            print(f"[Unexpected error] {e}")
            continue

        response_text = result["answer"]
        ctx_count = result["context_used"]
        _hr()
        print(f"Assistant ({provider.upper()}):\n")
        print(response_text)
        _hr()
        print(f"  Sources: {ctx_count} chunk(s) retrieved")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})


def main():
    parser = argparse.ArgumentParser(description="Graph RAG interactive chat")
    parser.add_argument("--provider", "-p", choices=["lmstudio", "claude"], default=None)
    args = parser.parse_args()
    run_chat(provider_override=args.provider)


if __name__ == "__main__":
    main()
