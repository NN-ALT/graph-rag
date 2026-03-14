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
from logging_config import setup_logging


def cmd_ingest(args):
    from ingestion.pipeline import ingest
    result = ingest(args.path, strategy=args.strategy)
    print(f"\nIngested: {result['title']}")
    print(f"  Document ID : {result['document_id']}")
    print(f"  Chunks      : {result['chunks']}")


def cmd_build_graph(args):
    from graph.builder import build_graph_from_all_documents
    result = build_graph_from_all_documents(force=args.force)
    if result["nodes"] == 0 and result["edges"] == 0:
        print("\nNothing to do — all chunks are already graph-indexed.")
        print("Use --force to rebuild from scratch.")
    else:
        print(f"\nGraph built:")
        print(f"  Nodes upserted : {result['nodes']}")
        print(f"  Edges upserted : {result['edges']}")


def cmd_create_index(args):
    from db.connection import get_conn
    from db.queries import create_vector_index, get_stats
    with get_conn() as conn:
        stats = get_stats(conn)
        embedding_count = stats.get("embeddings", 0)
        if embedding_count < 100:
            print(f"\nWarning: only {embedding_count} embeddings in the database.")
            print("IVFFlat index performs best with 1000+ rows.")
            print("Proceeding anyway...")
        print(f"\nBuilding IVFFlat vector index on {embedding_count} embeddings...")
        create_vector_index(conn)
    print("Vector index created (or already exists).")


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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("path", help="Path to .txt, .md, or .pdf file")
    p_ingest.add_argument("--strategy", default="sentence", choices=["sentence", "fixed"])

    # build-graph
    p_build = sub.add_parser("build-graph", help="Build graph from unindexed chunks")
    p_build.add_argument("--force", action="store_true",
                         help="Reprocess all chunks, not just new ones")

    # create-index
    sub.add_parser("create-index", help="Create IVFFlat vector index (run after bulk ingest)")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--hops", type=int, default=1)
    p_query.add_argument("--min-sim", type=float, default=0.4, dest="min_sim")

    sub.add_parser("stats", help="Show database statistics")
    sub.add_parser("models", help="List LM Studio loaded models")

    args = parser.parse_args()
    setup_logging(verbose=getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "build-graph": cmd_build_graph,
        "create-index": cmd_create_index,
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
