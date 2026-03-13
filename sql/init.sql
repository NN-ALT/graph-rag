-- ============================================================
-- Graph RAG — Database Initialization
-- Auto-executed on first container start by Docker entrypoint
-- ============================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'supabase_admin') THEN
    CREATE ROLE supabase_admin SUPERUSER LOGIN CREATEROLE CREATEDB REPLICATION BYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticator') THEN
    CREATE ROLE authenticator NOINHERIT LOGIN;
  END IF;
END
$$;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anon') THEN
    CREATE ROLE anon NOLOGIN NOINHERIT;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated NOLOGIN NOINHERIT;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'service_role') THEN
    CREATE ROLE service_role NOLOGIN NOINHERIT BYPASSRLS;
  END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title       TEXT NOT NULL,
    source      TEXT,
    content     TEXT NOT NULL,
    doc_type    TEXT DEFAULT 'text',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_order ON chunks(document_id, chunk_index);

CREATE TABLE IF NOT EXISTS embeddings (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_name  TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    embedding   VECTOR(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    label           TEXT NOT NULL,
    node_type       TEXT NOT NULL,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    properties      JSONB DEFAULT '{}',
    embedding       VECTOR(384),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(label, node_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_label ON graph_nodes(label);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type  ON graph_nodes(node_type);

CREATE TABLE IF NOT EXISTS graph_edges (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node_id  UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_node_id  UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL,
    weight          FLOAT DEFAULT 1.0,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    properties      JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source   ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target   ON graph_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relation ON graph_edges(relation_type);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_unique
    ON graph_edges(source_node_id, target_node_id, relation_type);

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, service_role;

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(384),
    match_count     INT DEFAULT 5,
    min_similarity  FLOAT DEFAULT 0.4
)
RETURNS TABLE (
    chunk_id        UUID,
    document_id     UUID,
    content         TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id            AS chunk_id,
        c.document_id,
        c.content,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM embeddings e
    JOIN chunks c ON c.id = e.chunk_id
    WHERE 1 - (e.embedding <=> query_embedding) >= min_similarity
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
