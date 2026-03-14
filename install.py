"""
Graph RAG — Interactive Installer
Run this after cloning the repo:  python install.py
"""

import os
import sys
import shutil
import subprocess
import platform
import secrets
import time
import base64
import hmac
import hashlib
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
PY  = sys.executable


def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 40)


def ok(msg: str):   print(f"  OK  {msg}")
def info(msg: str): print(f"  >>  {msg}")
def warn(msg: str): print(f"  !!  {msg}")


def ask(prompt: str, choices: list[str] | None = None, default: str = "") -> str:
    if choices:
        options = "/".join(c.upper() if c == default else c for c in choices)
        prompt = f"{prompt} [{options}]: "
    else:
        prompt = f"{prompt}: "
    while True:
        answer = input(prompt).strip()
        if not answer and default:
            return default.lower()
        if choices:
            if answer.lower() in [c.lower() for c in choices]:
                return answer.lower()
            print(f"  Please enter one of: {', '.join(choices)}")
        else:
            return answer


def ask_secret(prompt: str) -> str:
    """Ask for sensitive input without echoing (falls back to plain input)."""
    try:
        import getpass
        return getpass.getpass(f"{prompt}: ").strip()
    except Exception:
        return input(f"{prompt}: ").strip()


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    info("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if check and result.returncode != 0:
        warn(f"Command failed (exit {result.returncode})")
        if capture and result.stderr:
            print(result.stderr)
        sys.exit(1)
    return result


def cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def docker_running() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_jwt(payload: dict, secret: str) -> str:
    header_b = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body     = b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig      = b64url(hmac.new(secret.encode(), f"{header_b}.{body}".encode(), hashlib.sha256).digest())
    return f"{header_b}.{body}.{sig}"


# ── AI Configuration Questionnaire ────────────────────────────────────────────

def ask_ai_config() -> dict:
    """
    Questions 3–5: gather LLM provider, credentials, and chunking preferences.
    Returns a dict consumed by write_env().
    """
    config = {}

    # Q3 — LLM Provider
    print("\nQuestion 3 of 5 — LLM Provider")
    print("  lmstudio  : fully offline, uses a local model via LM Studio")
    print("  claude    : Anthropic API, requires an API key and internet access")
    provider = ask("  Choose provider", ["lmstudio", "claude"], default="lmstudio")
    config["llm_provider"] = provider
    ok(f"Provider: {provider}")

    # Q4a — LM Studio details
    if provider == "lmstudio":
        print("\nQuestion 4 of 5 — LM Studio")
        if not cmd_exists("lms") and not cmd_exists("LM Studio"):
            warn("LM Studio does not appear to be installed.")
            info("Download it from: https://lmstudio.ai/")
            info("You must have LM Studio running with a model loaded before using 'query' or 'chat'.")
        lms_url = ask(
            "  LM Studio server URL (press Enter for default)",
            default="http://localhost:1234/v1",
        ) or "http://localhost:1234/v1"
        lms_model = ask(
            "  Default model name shown in LM Studio (press Enter to skip)",
            default="local-model",
        ) or "local-model"
        config["lm_studio_url"]   = lms_url
        config["lm_studio_model"] = lms_model
        config["anthropic_api_key"] = ""
        config["claude_model"]      = "claude-sonnet-4-6"
        config["chars_per_token"]   = 4
        ok(f"LM Studio URL: {lms_url}")

    # Q4b — Claude details
    else:
        print("\nQuestion 4 of 5 — Claude API")
        api_key = ask_secret("  Enter your ANTHROPIC_API_KEY")
        if not api_key:
            warn("No API key entered. You can add it to .env later (ANTHROPIC_API_KEY=...).")
        print("  Available models:")
        print("    1) claude-sonnet-4-6   (recommended — fast and capable)")
        print("    2) claude-opus-4-6     (most capable, slower)")
        print("    3) claude-haiku-4-5-20251001   (fastest, lightweight)")
        model_choice = ask("  Select model", ["1", "2", "3"], default="1")
        model_map = {
            "1": "claude-sonnet-4-6",
            "2": "claude-opus-4-6",
            "3": "claude-haiku-4-5-20251001",
        }
        config["anthropic_api_key"] = api_key
        config["claude_model"]      = model_map[model_choice]
        config["lm_studio_url"]     = "http://localhost:1234/v1"
        config["lm_studio_model"]   = "local-model"
        config["chars_per_token"]   = 3   # Claude tokenizer is denser than most LM Studio models
        ok(f"Claude model: {config['claude_model']}")

    # Q5 — Chunking preferences
    print("\nQuestion 5 of 5 — Document Chunking")
    print("  Chunk size controls how text is split before embedding.")
    print("  Smaller chunks (256–512) = more precise retrieval.")
    print("  Larger chunks (512–1024) = more context per result.")
    use_defaults = ask("  Use default chunk settings? (size=512, overlap=64)", ["y", "n"], default="y")
    if use_defaults == "y":
        config["chunk_size"]    = 512
        config["chunk_overlap"] = 64
        ok("Using default chunk settings.")
    else:
        try:
            config["chunk_size"]    = int(ask("  Chunk size (characters)", default="512") or 512)
            config["chunk_overlap"] = int(ask("  Chunk overlap (characters)", default="64") or 64)
        except ValueError:
            warn("Invalid input — using defaults (512 / 64).")
            config["chunk_size"]    = 512
            config["chunk_overlap"] = 64
        ok(f"Chunk size: {config['chunk_size']}, overlap: {config['chunk_overlap']}")

    return config


# ── .env Writer ───────────────────────────────────────────────────────────────

def write_env(db_password: str, ai_config: dict,
              jwt_secret: str = "", anon_key: str = "", service_key: str = ""):
    env_path = os.path.join(ROOT, ".env")
    content = f"""# Generated by install.py — do NOT commit this file

POSTGRES_PASSWORD={db_password}
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD={db_password}

JWT_SECRET={jwt_secret}
ANON_KEY={anon_key}
SERVICE_ROLE_KEY={service_key}

LLM_PROVIDER={ai_config['llm_provider']}
LM_STUDIO_URL={ai_config['lm_studio_url']}
LM_STUDIO_MODEL={ai_config['lm_studio_model']}

ANTHROPIC_API_KEY={ai_config['anthropic_api_key']}
CLAUDE_MODEL={ai_config['claude_model']}

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384

CHUNK_SIZE={ai_config['chunk_size']}
CHUNK_OVERLAP={ai_config['chunk_overlap']}
CHARS_PER_TOKEN={ai_config['chars_per_token']}
"""
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)
    ok(f".env written to {env_path}")


def install_python_deps():
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    run([PY, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])
    ok("Python dependencies installed.")


def download_nlp_models():
    info("Downloading spaCy model (en_core_web_sm)...")
    run([PY, "-m", "spacy", "download", "en_core_web_sm"])
    info("Downloading NLTK data...")
    run([PY, "-c", "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"])
    info("Pre-caching sentence-transformers embedding model...")
    run([PY, "-c",
         "from sentence_transformers import SentenceTransformer; "
         "SentenceTransformer('all-MiniLM-L6-v2'); print('  Cached.')"])
    ok("NLP models ready.")


def register_mcp(detected_os: str):
    print()
    answer = ask("Register the Graph RAG MCP server with Claude Code?", ["y", "n"], default="y")
    if answer != "y":
        info("Skipped. To register later:")
        info(f"  claude mcp add graph-rag -s user -- {PY} mcp_server.py")
        return
    if not cmd_exists("claude"):
        warn("'claude' command not found.")
        info(f"  cd {ROOT}")
        info(f"  claude mcp add graph-rag -s user -- {PY} mcp_server.py")
        return
    run(["claude", "mcp", "add", "graph-rag", "-s", "user",
         "--", PY, os.path.join(ROOT, "mcp_server.py")])
    ok("Graph RAG MCP server registered with Claude Code.")


def print_final_summary(mode: str, ai_config: dict):
    provider = ai_config["llm_provider"]
    if provider == "claude":
        llm_note = f"Claude API ({ai_config['claude_model']})"
    else:
        llm_note = f"LM Studio ({ai_config['lm_studio_url']})"

    header("Installation Complete!")
    print(f"""
  Mode      : {mode}
  Provider  : {llm_note}
  Directory : {ROOT}

  Quick start:
    {PY} chat.py                          — start chatting
    {PY} main.py ingest <file>            — add a document
    {PY} main.py stats                    — check knowledge base
    {PY} main.py query "your question"    — ask without chat UI

  To switch LLM provider later, edit LLM_PROVIDER in .env:
    LLM_PROVIDER=lmstudio   — fully offline (LM Studio must be running)
    LLM_PROVIDER=claude     — Anthropic API (set ANTHROPIC_API_KEY in .env)

  See COMMANDS.txt for the full reference.
""")


def install_docker_path(detected_os: str, ai_config: dict, total_steps: int = 6):
    header("Installation via Docker")
    step(1, total_steps, "Generating secrets (.env)")
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        if ask(".env already exists. Regenerate?", ["y", "n"], default="n") == "y":
            _generate_secrets_docker(ai_config)
        else:
            ok("Keeping existing .env")
    else:
        _generate_secrets_docker(ai_config)
    step(2, total_steps, "Starting Docker containers (docker compose up -d)")
    run(["docker", "compose", "up", "-d"], check=True)
    info("Waiting 10 s for the database to initialize...")
    time.sleep(10)
    ok("Containers started.")
    step(3, total_steps, "Installing Python dependencies")
    install_python_deps()
    step(4, total_steps, "Downloading NLP & embedding models (one-time)")
    download_nlp_models()
    step(5, total_steps, "Claude Code MCP registration")
    register_mcp(detected_os)
    step(6, total_steps, "Verifying database connection")
    _verify_db()
    print_final_summary("Docker (Supabase stack)", ai_config)


def _generate_secrets_docker(ai_config: dict):
    db_pass    = secrets.token_urlsafe(24)
    jwt_secret = secrets.token_urlsafe(48)
    iat, exp   = 1641769200, 9999999999
    anon_key   = make_jwt({"role": "anon",         "iss": "supabase", "iat": iat, "exp": exp}, jwt_secret)
    svc_key    = make_jwt({"role": "service_role", "iss": "supabase", "iat": iat, "exp": exp}, jwt_secret)
    write_env(db_pass, ai_config, jwt_secret, anon_key, svc_key)


def _verify_db():
    try:
        result = subprocess.run(
            [PY, "-c",
             "import sys; sys.path.insert(0,'.'); "
             "from db.connection import get_conn; "
             "from db.queries import get_stats; "
             "conn_ctx = get_conn(); conn = conn_ctx.__enter__(); "
             "s = get_stats(conn); print('Documents:', s['documents'])"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode == 0:
            ok("Database connection successful.")
        else:
            warn("Could not connect to database. The containers may still be starting.")
            info(f"Try again in 30 s: {PY} main.py stats")
    except Exception as e:
        warn(f"Verification skipped: {e}")


def install_native_path(detected_os: str, ai_config: dict, total_steps: int = 7):
    header("Installation — Native PostgreSQL (no Docker)")
    step(1, total_steps, "Installing PostgreSQL + pgvector")
    _install_postgres_native(detected_os)
    step(2, total_steps, "Generating secrets (.env)")
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        if ask(".env already exists. Regenerate?", ["y", "n"], default="n") == "y":
            write_env(secrets.token_urlsafe(24), ai_config)
        else:
            ok("Keeping existing .env")
    else:
        write_env(secrets.token_urlsafe(24), ai_config)
    step(3, total_steps, "Initialising database schema")
    _init_native_db(detected_os)
    step(4, total_steps, "Installing Python dependencies")
    install_python_deps()
    step(5, total_steps, "Downloading NLP & embedding models (one-time)")
    download_nlp_models()
    step(6, total_steps, "Claude Code MCP registration")
    register_mcp(detected_os)
    step(7, total_steps, "Verifying database connection")
    _verify_db()
    print_final_summary("Native PostgreSQL", ai_config)


def _install_postgres_native(detected_os: str):
    if detected_os == "macos":
        if not cmd_exists("brew"):
            warn("Homebrew not found. Install it first: https://brew.sh")
            sys.exit(1)
        run(["brew", "install", "postgresql@15", "pgvector"])
        run(["brew", "services", "start", "postgresql@15"])
    elif detected_os == "linux":
        run(["sudo", "apt-get", "update", "-qq"])
        run(["sudo", "apt-get", "install", "-y", "-qq", "postgresql", "postgresql-contrib", "libpq-dev"])
        result = run(["sudo", "apt-get", "install", "-y", "-qq", "postgresql-16-pgvector"], check=False)
        if result.returncode != 0:
            _build_pgvector_linux()
        run(["sudo", "systemctl", "enable", "--now", "postgresql"])
    elif detected_os == "windows":
        if cmd_exists("winget"):
            run(["winget", "install", "--id", "PostgreSQL.PostgreSQL", "-e", "--silent"])
        else:
            warn("winget not available. Please install PostgreSQL manually.")
            input("Press Enter once PostgreSQL is installed...")
        warn("pgvector must be installed separately on Windows.")
        input("Press Enter once pgvector is installed...")
    ok("PostgreSQL ready.")


def _build_pgvector_linux():
    run(["sudo", "apt-get", "install", "-y", "-qq", "build-essential", "postgresql-server-dev-all", "git"])
    tmp = "/tmp/pgvector"
    if not os.path.exists(tmp):
        run(["git", "clone", "--depth", "1", "https://github.com/pgvector/pgvector.git", tmp])
    run(["make", "-C", tmp])
    run(["sudo", "make", "-C", tmp, "install"])


def _init_native_db(detected_os: str):
    db_password = ""
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("DB_PASSWORD="):
                db_password = line.split("=", 1)[1].strip()
                break
    if not db_password:
        warn("Could not read DB_PASSWORD from .env")
        sys.exit(1)
    sql_path = os.path.join(ROOT, "sql", "init.sql")
    psql = shutil.which("psql") or (r"C:\Program Files\PostgreSQL\17\bin\psql.exe" if detected_os == "windows" else "psql")
    set_pw_sql = f"ALTER USER postgres PASSWORD '{db_password}';"
    if detected_os in ("linux", "macos"):
        run(["sudo", "-u", "postgres", psql, "-c", set_pw_sql])
        subprocess.run(["sudo", "-u", "postgres", psql, "-f", sql_path], check=True)
    else:
        env_with_pw = os.environ.copy()
        env_with_pw["PGPASSWORD"] = db_password
        subprocess.run([psql, "-U", "postgres", "-c", set_pw_sql], check=False)
        subprocess.run([psql, "-U", "postgres", "-f", sql_path], env=env_with_pw, check=True)
    ok("Database schema initialised.")


def main():
    header("Graph RAG — Installer")
    print("""
  This installer will ask 5 questions then handle everything:
    1. Platform detection
    2. Docker availability
    3. LLM provider (LM Studio or Claude API)
    4. Provider credentials / connection details
    5. Document chunking preferences

  Then it will:
    - Set up PostgreSQL + pgvector (via Docker or natively)
    - Generate secure credentials (.env)
    - Install Python dependencies
    - Download NLP and embedding models
    - Optionally register the MCP server with Claude Code
""")

    # Q1 — Platform
    auto_os  = platform.system().lower()
    os_map   = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    detected = os_map.get(auto_os, "")

    print("Question 1 of 5 — Platform")
    print(f"  Auto-detected: {detected or 'unknown'}")
    choices_os = ["windows", "macos", "linux"]
    if detected:
        answer_os = ask("  Is this correct?  (Enter to confirm, or type to change)", choices_os, default=detected)
    else:
        answer_os = ask("  Select your platform", choices_os)
    ok(f"Platform: {answer_os}")

    # Q2 — Docker
    print("\nQuestion 2 of 5 — Docker")
    has_docker_cmd = cmd_exists("docker")
    docker_ok      = has_docker_cmd and docker_running()
    if docker_ok:
        info("Docker detected and running.")
    elif has_docker_cmd:
        info("Docker is installed but the daemon is not running.")
    else:
        info("Docker not found on this machine.")

    answer_docker = ask("  Install via Docker?", ["y", "n"], default="y" if docker_ok else "n")
    if answer_docker == "y" and not docker_ok:
        warn("Docker daemon is not reachable.")
        if ask("  Proceed anyway?", ["y", "n"], default="n") != "y":
            info("Please start Docker and re-run the installer.")
            sys.exit(0)

    # Q3–5 — AI configuration
    ai_config = ask_ai_config()

    print()
    os.chdir(ROOT)
    if answer_docker == "y":
        install_docker_path(answer_os, ai_config)
    else:
        install_native_path(answer_os, ai_config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Installation cancelled.")
        sys.exit(0)
