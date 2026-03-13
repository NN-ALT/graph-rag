"""
Graph RAG — CLI entry point

Commands:
    py main.py ingest <file_path>          Ingest a document into the knowledge base
    py main.py build-graph                 (Re)build graph from all stored documents
    py main.py query "<question>"          Ask a question against the knowledge base
    py main.py stats                       Show database statistics
    py main.py models                      Show active LLM provider and available models

LLM providers (set LLM_PROVIDER in .env):
    lmstudio   Fully offline — routes to LM Studio local server
    claude     Anthropic API — requires ANTHROPIC_API_KEY
"""

import argparse
import sys
import json


def cmd_ingest(args):
    from ingestion.pipeline import ingest
    result = ingest(args.path, strategy=args.strategy)
    print(f"\nIngested: {result['title']}")
    print(f"  Document ID : {result['document_id']}")
    print(f"  Chunks      : {result['chunks']}")


def cmd_build_graph(args):
    from graph.builder import build_graph_from_all_documents
    result = build_graph_from_all_documents()
    print(f"\nGraph built:")
    print(f"  Nodes upserted : {result['nodes']}")
    print(f"  Edges upserted : {result['edges']}")


def cmd_query(args):
    from rag.pipeline import answer
    print(f"\nQuerying: {args.question!r}")
    print(f"{'\u2500' * 60}")
    result = answer(args.question, k=args.k, graph_hops=args.hops, min_similarity=args.min_sim)
    print(result["answer"])
    print(f"\n{'\u2500' * 60}")
    print(f"Context chunks used: {result['context_used']}")


def cmd_stats(args):
    from db.connection import get_conn
    from db.queries import get_stats
    with get_conn() as conn:
        stats = get_stats(conn)
    print("\nDatabase stats:")
    for key, val in stats.items():
        print(f"  {key:<15} {val}")


def cmd_models(args):
    from llm.client import list_models, active_provider
    provider = active_provider()
    models = list_models()
    print(f"\nActive provider: {provider}")
    if models:
        print("Available models:")
        for m in models:
            print(f"  \u2022 {m}")
    else:
        if provider == "lmstudio":
            print("No models found. Is LM Studio running with a model loaded?")
        else:
            print("No models listed.")


def main():
    parser = argparse.ArgumentParser(
        description="Graph RAG — offline knowledge base with LM Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("path", help="Path to .txt, .md, or .pdf file")
    p_ingest.add_argument("--strategy", default="sentence", choices=["sentence", "fixed"])

    sub.add_parser("build-graph", help="Rebuild graph from all stored documents")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--hops", type=int, default=1)
    p_query.add_argument("--min-sim", type=float, default=0.4, dest="min_sim")

    sub.add_parser("stats", help="Show database statistics")
    sub.add_parser("models", help="List LM Studio loaded models")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "build-graph": cmd_build_graph,
        "query": cmd_query,
        "stats": cmd_stats,
        "models": cmd_models,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
