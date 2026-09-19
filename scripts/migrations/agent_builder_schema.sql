-- =============================================================================
-- Agent Builder schema replication
-- =============================================================================
-- Mirrors the tables defined in agent_builder_v1/src/db/models.py so the
-- Document Intelligence GCP Cloud SQL instance can host both services.
--
-- Source of truth: biz_2_bricks_v2/agent_builder_v1/src/db/models.py
-- Tables:
--   - agent_definitions
--   - agent_versions
--   - agent_runs
--
-- Safe to run multiple times (IF NOT EXISTS).
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- agent_definitions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_definitions (
    id              VARCHAR(36) PRIMARY KEY,
    organization_id VARCHAR(36) NOT NULL,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    mode            VARCHAR(32) NOT NULL,
    status          VARCHAR(32) NOT NULL DEFAULT 'draft',
    draft_payload   JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_definitions_organization_id
    ON agent_definitions (organization_id);

-- ---------------------------------------------------------------------------
-- agent_versions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_versions (
    id                  VARCHAR(36) PRIMARY KEY,
    agent_definition_id VARCHAR(36) NOT NULL REFERENCES agent_definitions(id),
    organization_id     VARCHAR(36) NOT NULL,
    version_number      INTEGER NOT NULL,
    graph_spec          JSONB NOT NULL,
    publish_notes       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_versions_agent_definition_id
    ON agent_versions (agent_definition_id);
CREATE INDEX IF NOT EXISTS ix_agent_versions_organization_id
    ON agent_versions (organization_id);

-- ---------------------------------------------------------------------------
-- agent_runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    id                VARCHAR(36) PRIMARY KEY,
    organization_id   VARCHAR(36) NOT NULL,
    agent_version_id  VARCHAR(36) NOT NULL REFERENCES agent_versions(id),
    status            VARCHAR(32) NOT NULL DEFAULT 'pending',
    input_payload     JSONB NOT NULL,
    output_payload    JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_runs_organization_id
    ON agent_runs (organization_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_agent_version_id
    ON agent_runs (agent_version_id);

COMMIT;
