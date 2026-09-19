# Biz2Bricks Platform v2 — Business Requirements Document (BRD)

**Document Intelligence + Enterprise Search SaaS with Agent Builder**

| Field | Value |
|---|---|
| Document ID | B2B-BRD-V2-001 |
| Version | 1.0 (Draft) |
| Author | VP Engineering, Biz2Bricks |
| Audience | Engineering team (implementation via Claude Code + Codex) |
| Status | For review and refinement |
| Related docs | `docs/hand_written/biz2bricks_platform_architecture.html`, `docs/hand_written/biz2bricks_hybrid_query_intelligence.html`, `CLAUDE.md`, `README.md` |

---

## Table of Contents

1. Executive Summary
2. Strategic Vision and Product Positioning
3. Personas, Roles, and Access Model
4. Core Use Cases and User Journeys
5. Product Scope — In/Out for v2
6. Functional Requirements
7. Non-Functional Requirements
8. System Architecture (3-App SaaS)
9. Data Model
10. API Contracts (Summary)
11. Event-Driven Architecture
12. Security, Privacy, and Compliance
13. Observability, SLOs, and Reliability
14. DevOps, Deployment, and Environments
15. Release Phasing and Roadmap
16. Success Metrics and Acceptance Criteria
17. Open Questions, Assumptions, and Risks
18. Appendix

---

## 1. Executive Summary

### 1.1 What We Are Building

Biz2Bricks v2 is a multi-tenant SaaS platform that unifies **document intelligence** (parsing, structured extraction, summarization, Q&A, business intelligence reports) with **enterprise search** across the customer's connected SaaS systems (Google Drive, SharePoint/OneDrive, Slack, Confluence, Jira, Notion, Gmail, Salesforce, HubSpot, Dropbox, Box). On top of this unified knowledge fabric, customers can build and deploy **custom AI agents** through a no-code/low-code Agent Builder.

Think of v2 as "Glean for SMBs, with document intelligence and an agent workbench built in." Unlike Glean's enterprise-only focus, Biz2Bricks v2 is designed for 10–500 person organizations with:

- **Self-service onboarding** (<5 minutes from signup to first extracted document).
- **Industry templates** (Food Manufacturing, Recruitment, Retail, Real Estate to start) that pre-configure document types, dashboards, and prompts.
- **Usage-based tiered pricing** (Free → Pro → Enterprise) with quota enforcement.
- **Per-tenant data isolation** via row-level security, per-tenant GCS prefixes, and per-tenant Gemini File Search stores.

### 1.2 Why Now

Biz2Bricks v1 (the current `doc_intelligence_ai_v3.0`, `doc_intelligence_backend_api_v2.0`, `agent_builder_v1`, `document_intelligence_fe_v2` stack) has proved the core AI pipelines: document parsing, structured extraction, summaries/FAQs/questions, BI reports, bulk processing, and usage tracking. It is not yet a productized SaaS — identity, billing, admin tooling, connector integrations, unified search, and the agent builder are fragmentary or absent. v2 closes those gaps and repackages the stack as a three-app SaaS.

### 1.3 Three-App Topology

The v2 platform is deployed as three independently releasable applications sharing the same data plane:

| App | Purpose | Tech | Repo (suggested) |
|---|---|---|---|
| **AI Backend** | Agents (Document, Sheets, Extractor, Report, Search, Orchestrator, Custom), LLM orchestration, RAG, hybrid query router, text-to-SQL, connector sync workers, embedding pipelines | Python 3.12, FastAPI, LangGraph, LangChain, Gemini 3 / OpenAI, DuckDB, LlamaParse | `biz2bricks-ai-v2` |
| **Admin Backend** | Identity, RBAC, orgs/tenants, billing, plans/quotas, SSO, audit, connectors config, agent registry/versioning, knowledge graph management, admin console APIs | Python 3.12, FastAPI, SQLAlchemy 2.0 async, Stripe, Auth0/WorkOS | `biz2bricks-admin-v2` |
| **Frontend** | End-user web app + Admin console, unified chat, dashboards, document viewer, agent builder UI, connector config, billing portal | Next.js 15 (App Router), React 19, Tailwind, shadcn/ui, Framer Motion, TanStack Query | `biz2bricks-fe-v2` |

A thin **API Gateway / BFF** (inside the Next.js app's route handlers, or a standalone Cloud Run service) sits in front of both backends to handle tenant-context injection, JWT validation, rate limiting, and request routing.

### 1.4 Key Differentiators vs. v1

1. **Unified Enterprise Search** across uploaded documents *and* 5–10 SaaS connectors. v1 only indexes uploaded files.
2. **First-class Agent Builder** — ship the no-code agent designer as a GA product pillar, not a science project.
3. **Productized Admin Backend** — separate service for identity, billing, and tenant ops. v1 co-mingles this in the AI backend.
4. **Hybrid Query Router** — a single natural-language entrypoint that auto-routes across structured SQL, semantic RAG, connector content, and agent tools.
5. **Knowledge Graph** — per-tenant entity graph (people, docs, projects, accounts) derived from connector metadata, usable by agents for grounded reasoning.
6. **True SSO + SCIM** — WorkOS or Auth0 integration for enterprise IdPs (Okta, Azure AD, Google Workspace).
7. **Modern Observability** — OpenTelemetry traces end-to-end, per-tenant cost attribution, LLM eval harness.
8. **Compliance Posture** — SOC 2 Type II ready, GDPR-compliant, regional data residency hooks.

---

## 2. Strategic Vision and Product Positioning

### 2.1 Vision Statement

> **Every SMB should be able to ask any question about any of their information — uploaded or scattered across SaaS tools — and get a trustworthy, cited answer in seconds. Every workflow that reads, classifies, or extracts information from those documents should be runnable as a self-serve agent.**

### 2.2 Market Positioning

| Competitor | Category | Why we win with SMBs |
|---|---|---|
| Glean | Enterprise search | Too expensive; no doc intelligence; no agent builder GA; no SMB onboarding. |
| Notion AI / Mem | Knowledge AI | No structured extraction; shallow connectors; no agent workbench. |
| Microsoft Copilot | Enterprise AI | Locked to Microsoft tenant; no industry templates; not usable cross-stack. |
| AirOps / Relevance AI | Agent builders | Thin on unified search and connector coverage; no doc intelligence depth. |
| Hyperscience / Docsumo | Extraction | Narrow (extraction only); no search or agents. |

Biz2Bricks v2 sits at the **intersection of four categories**: document intelligence, enterprise search, agent builder, and vertical BI. No competitor combines all four at the SMB price point.

### 2.3 North-Star Metric

**Weekly Active Agents Per Tenant (WAA/T)** — the number of distinct agent/workflow executions per tenant per week. This measures how deeply the platform is embedded in customer operations, not just whether they log in.

### 2.4 Guiding Principles

1. **Config over code.** Industry-specific behavior is data, not branches in code.
2. **Multi-tenant by default.** Every request, row, object, and model call carries a tenant id.
3. **LLM-agnostic.** Gemini, OpenAI, and Anthropic are swappable per agent/route via config.
4. **Grounded by construction.** Every LLM answer that touches customer data MUST cite sources.
5. **Observability is a feature.** If we cannot explain a response, it is a bug.
6. **Secure defaults.** Least privilege, encrypted at rest and in transit, audited.
7. **Self-serve first.** Any workflow a human at Biz2Bricks does during onboarding should become a self-serve flow within two quarters.

---

## 3. Personas, Roles, and Access Model

### 3.1 External Personas (Customer Side)

| Persona | Description | Primary goals | Key surfaces |
|---|---|---|---|
| **Tenant Owner** | Founder/CxO of SMB that signs up. | Sign up, pick template, invite team, monitor ROI. | Onboarding, billing, team mgmt, dashboard. |
| **Ops/Knowledge Worker** | Daily user: processes documents, asks questions. | Upload docs, extract fields, search, chat, run agents. | Chat, search, documents, extraction, reports. |
| **Analyst** | Builds custom reports and queries. | Build BI reports, write NL queries, export. | BI reports, Sheets agent, dashboards. |
| **Automation Builder** | Power user who creates agents. | Assemble agents from tools + prompts, test, publish. | Agent Builder, agent runs. |
| **IT/Security Admin** | Integrator, reviewer. | SSO setup, connector auth, audit logs, RBAC. | Admin console, audit, SSO, connectors. |
| **Viewer/Exec** | Consumes dashboards, answers. | See reports, drill down, export. | Dashboards, chat. |

### 3.2 Internal Personas (Biz2Bricks Side)

| Persona | Description | Surface |
|---|---|---|
| **Platform Admin** | Biz2Bricks staff managing all tenants. | Superadmin console: tier mgmt, tenant lifecycle, usage investigation. |
| **Customer Success** | Helps with onboarding, troubleshoots. | Read-mostly tenant view, impersonation with audit. |
| **Billing Ops** | Manages Stripe/invoice issues. | Billing subsection of admin. |

### 3.3 RBAC Role Matrix

Roles are defined at the **organization** level. A user can belong to multiple organizations with different roles.

| Capability | Owner | Admin | Member | Viewer | Auditor |
|---|---|---|---|---|---|
| Upload / parse documents | ✓ | ✓ | ✓ | ✗ | ✗ |
| Run chat / search | ✓ | ✓ | ✓ | ✓ | ✓ |
| Run extractions | ✓ | ✓ | ✓ | ✗ | ✗ |
| Build agents | ✓ | ✓ | ✓ | ✗ | ✗ |
| Publish agents (org-wide) | ✓ | ✓ | ✗ | ✗ | ✗ |
| Configure connectors | ✓ | ✓ | ✗ | ✗ | ✗ |
| Manage users / invite | ✓ | ✓ | ✗ | ✗ | ✗ |
| Manage billing | ✓ | ✗ | ✗ | ✗ | ✗ |
| Change SSO / IdP | ✓ | ✓ | ✗ | ✗ | ✗ |
| Read audit logs | ✓ | ✓ | ✗ | ✗ | ✓ |
| Delete organization | ✓ | ✗ | ✗ | ✗ | ✗ |

**Platform-level roles** (Biz2Bricks staff): `platform_superadmin`, `platform_support`, `platform_billing`, `platform_readonly`.

### 3.4 Resource-Level Permissions

Beyond role-based access, every **Folder**, **Agent**, **Knowledge Source**, and **Saved Query** has optional ACLs:

- `private` (owner only),
- `team:<team_id>` (specified team),
- `org` (everyone in org with the base role),
- `public_share_link` (tokenized public URL; view-only; expires).

Agents additionally carry a `publish_scope` attribute (`draft` | `private` | `org` | `marketplace_private` | `marketplace_public`) — only `marketplace_public` is gated behind platform review in v2.

---

## 4. Core Use Cases and User Journeys

### 4.1 Primary Use Cases

| UC-# | Title | Primary actor | Summary |
|---|---|---|---|
| UC-01 | Self-service onboarding | Tenant Owner | Signup → industry pick → template provision → first upload → "wow" moment. |
| UC-02 | Unified search | Ops Worker | Asks NL question, gets grounded answer spanning uploaded docs + connectors. |
| UC-03 | Structured extraction at scale | Ops Worker | Uploads batch (e.g., 200 invoices); receives CSV/XLSX with extracted fields. |
| UC-04 | Content generation | Ops Worker | Summaries, FAQs, comprehension questions from long docs. |
| UC-05 | BI report generation | Analyst | Pick folder, report type, date range → PDF/XLSX + dashboard tiles. |
| UC-06 | Hybrid Q&A | Analyst | "Invoices > $10K that mention penalties" — combines SQL + RAG. |
| UC-07 | Build a custom agent | Automation Builder | Assemble tools, prompts, schema; test; publish org-wide. |
| UC-08 | Run an agent on a schedule | Automation Builder | E.g., "every Monday summarize last week's new contracts in Drive." |
| UC-09 | Connector configuration | IT Admin | OAuth Google Drive, SharePoint, Slack; pick folders/channels; schedule sync. |
| UC-10 | SSO / SCIM setup | IT Admin | Point IdP at Biz2Bricks SAML/OIDC; auto-provision users. |
| UC-11 | Admin / audit | IT Admin, Owner | Review audit trail; revoke access; view usage and costs. |
| UC-12 | Billing self-serve | Owner | Upgrade plan, see usage, add seats, download invoices. |
| UC-13 | Platform admin | Platform Staff | Investigate tenant usage anomaly; adjust quota; impersonate with audit. |

### 4.2 Journey: UC-01 Self-Service Onboarding (target ≤5 min)

```
Step 1  Landing page → Sign up (email or Google SSO)              30s
Step 2  Verify email (magic link) → land on /onboard               30s
Step 3  Pick industry template (Food/Recruit/Retail/RE)            15s
Step 4  Enter org details (name, size, timezone)                   30s
Step 5  Invite teammates (optional, skippable)                     20s
Step 6  (Optional) Connect Google Drive / M365                     60s
Step 7  Upload first document OR pick sample                       30s
Step 8  Watch extraction happen live, see dashboard populate      60s
Step 9  Land on /dashboard with widgets + "next steps" checklist   -
```

Measured as a funnel in analytics; target ≥60% completion from Step 1 to Step 8.

### 4.3 Journey: UC-02 Unified Search

```
User types: "What's the total spend with Acme Corp this quarter
             and does their contract have a cap?"

→ Hybrid Query Router classifies as HYBRID (SQL + RAG)
→ SQL path: aggregates invoices from extracted_data WHERE vendor='Acme' AND quarter='Q2'
→ RAG path: semantic search in 'contracts' file store for 'Acme' + 'cap'
→ Connector path (if Slack/Drive connected): search for recent Acme messages/files
→ Synthesizer (Gemini 3 Flash) merges: quantitative answer + contract quotes + source cards
→ UI renders tabbed response: Answer | SQL | Sources | Connector Hits
```

### 4.4 Journey: UC-07 Build a Custom Agent

```
Agent Builder UI:
1. "New Agent" → pick template (blank | summarizer | extractor | router | reviewer)
2. Configure:
   - Name, description, avatar
   - Input schema (JSON Schema builder)
   - Tools (multi-select from tool registry: file_search, sql_query,
            gmail_send, slack_post, extract_schema, web_fetch, ...)
   - Knowledge sources (folders, file search stores, connector scopes)
   - Prompt (system message, with variable placeholders)
   - Model (gemini-3-flash | gpt-5.1 | claude-sonnet-4 | ...)
   - Guardrails (PII strategy, max tokens, max tool calls, timeout)
3. Test tab: send sample inputs, see traces (prompt, tool calls, tokens, cost, latency)
4. Eval tab: upload or define test cases, assert on output
5. Deploy: draft → private (owner) → org (shared) → trigger (schedule | webhook | chat)
6. Version: every publish creates an immutable version; rollback supported
```

---

## 5. Product Scope — In / Out for v2

### 5.1 IN SCOPE (v2 GA)

**Core capabilities (evolved from v1):**
- Multi-tenant identity and org management with SSO (SAML/OIDC) and SCIM.
- Document ingestion (upload, bulk upload, GCS-trigger ingestion).
- Document parsing with Gemini and LlamaParse (OCR, handwriting, tables).
- Structured extraction (field analysis, schema generation, record extraction).
- Content generation (summaries, FAQs, comprehension questions).
- BI report generation (expense, vendor analysis, invoice reconciliation, etc.).
- Sheets analysis (Excel/CSV natural language Q&A via DuckDB).
- Semantic search within uploaded documents (Gemini File Search).
- Conversational chat with citations.
- Session + conversation memory.

**New for v2:**
- **Unified Enterprise Search** (Glean-lite) across 5–10 SaaS connectors.
- **Connector framework** (OAuth, scheduled sync, delta/event sync, redaction rules).
- **Hybrid Query Router** (SQL + RAG + connector + tool-call routing).
- **Text-to-SQL engine** over per-tenant dynamic schemas (JSONB + custom fields).
- **Agent Builder** (no-code builder, tool/MCP registry, versioning, evals, deploys, schedules).
- **Knowledge Graph** per tenant (entities, relations, used by agents and search ranking).
- **Admin Backend** (separate service: identity, RBAC, billing, quota, audit, tenant lifecycle).
- **Billing** via Stripe (plans, seats, metered add-ons).
- **Platform Superadmin console**.
- **Industry templates** (Food, Recruitment, Retail, Real Estate).
- **Custom field / schema builder** (tenant-editable document types).
- **Observability** (OpenTelemetry, LLM eval harness, per-tenant cost attribution).

**Connectors — v2 GA targets (launch with 5, follow with 5 within 90 days):**
1. Google Drive (GA at launch)
2. Microsoft OneDrive / SharePoint (GA at launch)
3. Slack (GA at launch)
4. Gmail (GA at launch)
5. Confluence (GA at launch)
6. Jira (fast follow)
7. Notion (fast follow)
8. Dropbox / Box (fast follow)
9. Salesforce (fast follow)
10. HubSpot (fast follow)

### 5.2 OUT OF SCOPE (v2)

- Public agent marketplace with third-party publishers (v3).
- Revenue share / payouts for marketplace authors (v3).
- On-prem / self-hosted deployment (Enterprise SKU may offer dedicated tenant on GCP; not self-hosted).
- Non-English OCR/extraction beyond what Gemini/LlamaParse provide out of the box.
- Mobile-native apps (responsive web only).
- Voice/phone interface.
- Real-time collaborative editing of documents.
- Full competitive replacement for vertical ERP systems (NetSuite, etc.).
- Consumer (non-business) use cases.

### 5.3 Explicit Non-Goals

- We are **not** a generic chatbot for the open web.
- We are **not** a code generation / developer assistant.
- We are **not** competing head-on with Glean at the 10,000+ seat tier; we are the SMB/mid-market alternative.

---

## 6. Functional Requirements

Requirements are grouped by capability area. Each requirement is testable and prefixed `FR-<area>-<n>`. Priority: **P0** (must ship for v2 GA), **P1** (ship within 90 days post-GA), **P2** (nice-to-have, deferable).

### 6.1 Identity, Authentication, and SSO

| ID | Priority | Requirement |
|---|---|---|
| FR-ID-01 | P0 | Users sign up with email + password (bcrypt/argon2). |
| FR-ID-02 | P0 | Users can sign up / sign in with Google OAuth and Microsoft OAuth. |
| FR-ID-03 | P0 | Email verification is required before first paid action (upload, invite). |
| FR-ID-04 | P0 | Passwordless magic-link login is supported. |
| FR-ID-05 | P0 | MFA via TOTP and WebAuthn is available to all users; enforceable org-wide by Admin. |
| FR-ID-06 | P0 | Enterprise SSO via SAML 2.0 and OIDC is configurable per org (domain-bound). |
| FR-ID-07 | P0 | SCIM 2.0 provisioning/deprovisioning is supported for enterprise tenants. |
| FR-ID-08 | P0 | JWT access tokens (15 min TTL) + refresh tokens (30 days TTL, rotating). |
| FR-ID-09 | P0 | Sessions are revocable per-user and org-wide from admin console. |
| FR-ID-10 | P0 | Impersonation is available to `platform_superadmin` with audit (reason required). |
| FR-ID-11 | P1 | Domain claim: verified domain auto-adds new signups to matching org. |
| FR-ID-12 | P1 | Service accounts (API keys scoped to an org) with configurable scopes. |

**Implementation note:** Use WorkOS or Auth0 for SSO/SCIM so we do not reimplement SAML. Keep first-party JWT for API auth.

### 6.2 Organization / Tenant Management

| ID | Priority | Requirement |
|---|---|---|
| FR-ORG-01 | P0 | A user can create one or more organizations (tenants). Each has a globally unique slug. |
| FR-ORG-02 | P0 | An org has: name, slug, industry_template_id, timezone, default_language, logo, company_size, data_region. |
| FR-ORG-03 | P0 | Inviting a user by email: pending invite, 7-day expiry, resendable, revocable. |
| FR-ORG-04 | P0 | A user can belong to multiple orgs; UI switcher in top nav. |
| FR-ORG-05 | P0 | Transferring ownership (Owner → another Owner) with email confirmation. |
| FR-ORG-06 | P0 | Org deletion: soft-delete + 30-day hold before hard delete; exports offered first. |
| FR-ORG-07 | P0 | Teams within an org: named groups of users for ACLs on folders/agents. |
| FR-ORG-08 | P1 | Data region selection at org creation (US, EU, APAC), pinned immutably. |
| FR-ORG-09 | P1 | Org-level IP allowlist for login / API access. |

### 6.3 Billing and Subscriptions

| ID | Priority | Requirement |
|---|---|---|
| FR-BIL-01 | P0 | Plans: **Free**, **Pro**, **Business**, **Enterprise** (contract). |
| FR-BIL-02 | P0 | Integration with Stripe (customer, subscription, invoice, webhook). |
| FR-BIL-03 | P0 | Metered usage: tokens, parsed pages, file search queries, storage, connector sync events. |
| FR-BIL-04 | P0 | Quota enforcement is **pre-check** at request time; returns 429 with retry-after. |
| FR-BIL-05 | P0 | Usage dashboard visible to Owner/Admin: current period, projection, breakdown by feature. |
| FR-BIL-06 | P0 | Plan upgrades/downgrades: prorated; takes effect immediately on upgrade, end-of-period on downgrade. |
| FR-BIL-07 | P0 | Seats are metered (unique active users/month); Business & Enterprise have seat tiers. |
| FR-BIL-08 | P0 | Invoice PDF download; billing history. |
| FR-BIL-09 | P0 | Grace period: 7 days past due before feature-gating; 30 days before account suspension. |
| FR-BIL-10 | P1 | Committed-use discounts (annual plan toggle). |
| FR-BIL-11 | P1 | Usage budget alerts at 50/80/100% with email notifications. |
| FR-BIL-12 | P2 | Customer-facing cost attribution by feature/agent (FinOps-like). |

**Default tier limits (initial):**

| Feature | Free | Pro ($49/mo) | Business ($149/mo) | Enterprise |
|---|---|---|---|---|
| Users | 3 | 10 | 50 | Unlimited |
| Docs / month | 50 | 1,000 | 10,000 | Unlimited / contract |
| Extraction fields | 10 / doc | 50 / doc | 100 / doc | Unlimited |
| Chat queries / mo | 100 | 2,000 | 20,000 | Unlimited |
| Connectors | 1 | 3 | 10 | Unlimited |
| Agents published | 1 | 10 | 50 | Unlimited |
| Storage | 1 GB | 25 GB | 250 GB | Custom |
| Token budget / mo | 50K | 1M | 10M | Negotiated |
| SSO | ✗ | ✗ | ✓ | ✓ |
| SCIM | ✗ | ✗ | ✗ | ✓ |
| Data region choice | ✗ | ✗ | ✓ | ✓ |
| Custom models (BYO key) | ✗ | ✗ | ✓ | ✓ |
| Dedicated support | email | email | chat+email | CSM |

### 6.4 Document Ingestion

| ID | Priority | Requirement |
|---|---|---|
| FR-ING-01 | P0 | Supported formats: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TXT, RTF, HTML, images (JPG, PNG, GIF, BMP, TIFF, WebP). |
| FR-ING-02 | P0 | Single upload up to 50 MB (Pro), 250 MB (Enterprise). |
| FR-ING-03 | P0 | Bulk upload: drag-and-drop multi-file or ZIP; concurrent processing ≥3 per job; ≥10 per folder per job (configurable). |
| FR-ING-04 | P0 | Upload via API with signed URLs; support resumable uploads ≥100 MB. |
| FR-ING-05 | P0 | GCS-trigger ingestion: uploads to a designated prefix start ingestion via Pub/Sub event. |
| FR-ING-06 | P0 | Folder hierarchy per tenant; documents live in folders; folders have ACLs. |
| FR-ING-07 | P0 | Duplicate detection via SHA-256 hash; user prompted to replace or skip. |
| FR-ING-08 | P0 | Virus scanning (ClamAV or Google Cloud Storage content scanning) on upload; infected files quarantined. |
| FR-ING-09 | P0 | PII detection on ingestion (configurable per org: redact | mask | hash | block | off). |
| FR-ING-10 | P1 | Email-to-upload: unique email address per folder; attachments parsed. |
| FR-ING-11 | P1 | Connector-sourced documents appear in a read-only virtual folder (not copied). |

### 6.5 Parsing and OCR

| ID | Priority | Requirement |
|---|---|---|
| FR-PARSE-01 | P0 | Default parser: Gemini 3 Flash (fast, multimodal). |
| FR-PARSE-02 | P0 | Fallback parser: LlamaParse (for complex tables / handwriting). |
| FR-PARSE-03 | P0 | Parser is configurable per org (`gemini` \| `llamaparse` \| `auto`). |
| FR-PARSE-04 | P0 | Output: Markdown + structured page metadata (page count, language, tables flagged, images). |
| FR-PARSE-05 | P0 | Cache: if content hash already parsed, skip re-parse. |
| FR-PARSE-06 | P0 | Parsed content is stored in GCS at `gs://{bucket}/orgs/{org_id}/parsed/{doc_id}.md`. |
| FR-PARSE-07 | P1 | Table extraction: tables produced as JSON alongside Markdown. |
| FR-PARSE-08 | P1 | Language detection with confidence; non-English routed to language-aware pipeline. |
| FR-PARSE-09 | P2 | Custom parser plug-ins (BYO parser service via signed webhook). |

### 6.6 Structured Extraction

| ID | Priority | Requirement |
|---|---|---|
| FR-EXT-01 | P0 | Field analysis: given a document, produce a list of candidate fields with types and confidence. |
| FR-EXT-02 | P0 | Schema generation: user selects fields → system produces JSON Schema template (versioned, named). |
| FR-EXT-03 | P0 | Extraction: given a document (or folder) and a schema, produce structured records. |
| FR-EXT-04 | P0 | Line-item extraction for invoice/receipt document types (array of items). |
| FR-EXT-05 | P0 | Validation rules per schema (required, regex, numeric bounds, cross-field, e.g., `total == sum(line_items.total)`). |
| FR-EXT-06 | P0 | Low-confidence extractions are flagged and can be human-reviewed (approval UI). |
| FR-EXT-07 | P0 | Bulk extraction over a folder with job tracking and partial-failure retry. |
| FR-EXT-08 | P0 | Extracted records are queryable (UI table + API) and exportable (CSV, JSON, XLSX). |
| FR-EXT-09 | P0 | Custom fields can be added to any system document type without code deploy. |
| FR-EXT-10 | P1 | Schema versioning: historical records remain tied to the schema version used. |
| FR-EXT-11 | P1 | Reviewer workflow: assign reviewer, status (`pending` / `approved` / `rejected`), audit trail. |
| FR-EXT-12 | P2 | Active learning: corrections feed back into prompt refinement (per tenant). |

### 6.7 Content Generation

| ID | Priority | Requirement |
|---|---|---|
| FR-GEN-01 | P0 | Generate **summary** (configurable max words; default 500, range 50–2000). |
| FR-GEN-02 | P0 | Generate **FAQs** (count 1–50, default 10, JSON format). |
| FR-GEN-03 | P0 | Generate **comprehension questions** (easy/medium/hard, count 1–100, default 10). |
| FR-GEN-04 | P0 | Generate **all** (summary + FAQs + questions) in one call; parallelized. |
| FR-GEN-05 | P0 | Results persist in GCS (`generated/`) and PostgreSQL with SHA-256 content key (cache). |
| FR-GEN-06 | P0 | Regeneration forces cache miss and creates new version. |
| FR-GEN-07 | P1 | Templated prompts per industry (e.g., "FAQ for food-safety audit") with preview. |
| FR-GEN-08 | P2 | User-definable generator types (e.g., "action items", "risk register"). |

### 6.8 Intelligence (BI) Reports

| ID | Priority | Requirement |
|---|---|---|
| FR-BI-01 | P0 | Report types (v2 seed): Expense Summary, Vendor Analysis, Invoice Reconciliation, Cash Flow, Spend Trends. |
| FR-BI-02 | P0 | Report takes folder + date range + options; emits PDF, XLSX, and JSON. |
| FR-BI-03 | P0 | Charts: bar, line, pie, category breakdown, monthly trend. |
| FR-BI-04 | P0 | Insights: LLM-generated narrative insights with citations to source records. |
| FR-BI-05 | P0 | Dashboards: configurable per template (widgets: KPI tiles, charts, tables). |
| FR-BI-06 | P1 | Scheduled reports (daily/weekly/monthly) delivered via email and Slack. |
| FR-BI-07 | P1 | Drill-down from dashboard widget to underlying records/documents. |
| FR-BI-08 | P2 | Custom report builder (user-defined queries + visualizations). |

### 6.9 Sheets Analysis

| ID | Priority | Requirement |
|---|---|---|
| FR-SHE-01 | P0 | Upload Excel (XLSX/XLS) or CSV; preview columns and rows. |
| FR-SHE-02 | P0 | Natural language queries (SheetsAgent + DuckDB). |
| FR-SHE-03 | P0 | Cross-file queries (join multiple sheets by user intent). |
| FR-SHE-04 | P0 | Statistical analysis: summary, correlation, trends, outliers, quality. |
| FR-SHE-05 | P0 | SQL injection protection (blocked keywords); only SELECT. |
| FR-SHE-06 | P0 | Result caching per session, LRU file cache. |
| FR-SHE-07 | P1 | Chart suggestions with auto-rendering. |
| FR-SHE-08 | P1 | Save queries; re-run on new uploads. |

### 6.10 Unified Enterprise Search (Glean-lite)

| ID | Priority | Requirement |
|---|---|---|
| FR-SRCH-01 | P0 | Single search box: returns results spanning uploaded docs + all connected sources. |
| FR-SRCH-02 | P0 | Search modes: semantic, keyword, hybrid (default: hybrid). |
| FR-SRCH-03 | P0 | Filters: source (connector), document type, folder, owner, date, file type, team. |
| FR-SRCH-04 | P0 | Faceted result UI with source badge, last-modified, preview snippet, citation. |
| FR-SRCH-05 | P0 | Per-user permission enforcement: search respects both Biz2Bricks ACLs and source-system permissions (Drive sharing, Slack channel membership). |
| FR-SRCH-06 | P0 | "Ask" toggle: switch from retrieval to generative answer with citations. |
| FR-SRCH-07 | P0 | Search analytics: query log, CTR, zero-result queries per org (for tuning). |
| FR-SRCH-08 | P1 | Saved searches + alert-on-new-match. |
| FR-SRCH-09 | P1 | People search: find SMEs by topic (using knowledge graph). |
| FR-SRCH-10 | P1 | Recent / recommended / trending sections on search home. |
| FR-SRCH-11 | P1 | Natural-language follow-ups ("show only PDFs", "from last week"). |
| FR-SRCH-12 | P2 | Personal knowledge card (per user): their owned docs, recent activity. |

### 6.11 Connectors Framework

| ID | Priority | Requirement |
|---|---|---|
| FR-CONN-01 | P0 | OAuth 2.0 flow for each connector; refresh tokens stored encrypted in Secret Manager. |
| FR-CONN-02 | P0 | Admin chooses scopes (folders/channels/spaces) to sync per connector. |
| FR-CONN-03 | P0 | Initial **full sync** on connect; **incremental/delta sync** on schedule (configurable cadence: 15 min / hourly / daily). |
| FR-CONN-04 | P0 | Webhooks / push subscriptions where available (Drive changes, Slack events) for near-real-time updates. |
| FR-CONN-05 | P0 | Document content is parsed, embedded, and indexed into per-tenant File Search stores. |
| FR-CONN-06 | P0 | ACL mirroring: each document carries the connector's original ACL (list of user emails / groups). |
| FR-CONN-07 | P0 | Per-user search respects those ACLs at query time (LCL — Least-Common-denominator-based access, computed from claims). |
| FR-CONN-08 | P0 | Redaction rules (regex / dictionary) applied before indexing. |
| FR-CONN-09 | P0 | Sync health dashboard per connector: last sync, error rate, skipped docs, reasons. |
| FR-CONN-10 | P0 | Disconnect: pauses sync; optional data purge on disconnect. |
| FR-CONN-11 | P1 | Rate-limit backoff for source APIs with jitter. |
| FR-CONN-12 | P1 | Cross-region data residency (EU-origin docs stay in EU index). |

**Connector module contract** (every connector implements):

```python
class Connector(Protocol):
    connector_id: str                                    # e.g. "google_drive"
    display_name: str
    auth_method: Literal["oauth2", "api_key", "saml"]
    async def authenticate(org_id, user_id) -> AuthResult
    async def list_scopes(org_id) -> list[Scope]         # e.g. folders, channels
    async def full_sync(org_id, scope) -> SyncReport     # emits DocumentEvent stream
    async def delta_sync(org_id, scope, cursor) -> SyncReport
    async def fetch_document(org_id, external_id) -> ParsedDoc
    async def permissions_for(external_id) -> ACL
    async def disconnect(org_id) -> None
```

### 6.12 Hybrid Query Router and Chat

| ID | Priority | Requirement |
|---|---|---|
| FR-HQR-01 | P0 | A natural-language question is classified into one of: `sql_only`, `rag_only`, `connector_only`, `hybrid`, `tool_agent`. |
| FR-HQR-02 | P0 | Router returns `RouterDecision{route, reasoning, sub_questions, confidence}` using Gemini 3 Flash structured output. |
| FR-HQR-03 | P0 | Text-to-SQL engine generates safe PostgreSQL against per-tenant dynamic schema (JSONB + custom fields). |
| FR-HQR-04 | P0 | SQL validation via `sqlglot` AST; only SELECT; tenant_id filter mandatory; 100-row default limit; 5s timeout. |
| FR-HQR-05 | P0 | RAG path queries per-tenant Gemini File Search stores + connector indexes, scoped by ACLs. |
| FR-HQR-06 | P0 | Synthesizer merges paths into a final answer with citations; streamed via SSE. |
| FR-HQR-07 | P0 | Observability: router decisions, SQL queries, latencies, tokens, and citations are logged per query. |
| FR-HQR-08 | P0 | Session memory (30 min) for chat; long-term memory (summaries) persisted. |
| FR-HQR-09 | P1 | Suggested follow-ups generated after each answer. |
| FR-HQR-10 | P1 | Auto-visualization suggestion for SQL result sets (line/bar/table). |

### 6.13 Agent Builder

Treated as a first-class product surface.

| ID | Priority | Requirement |
|---|---|---|
| FR-AGB-01 | P0 | Agent definition: `name`, `description`, `avatar`, `input_schema`, `output_schema`, `tools[]`, `knowledge_sources[]`, `system_prompt`, `model`, `guardrails`, `version`. |
| FR-AGB-02 | P0 | Tool registry: built-in tools (search, sql_query, file_store_query, extract_with_schema, web_fetch, summarize, generate_report, send_email, post_slack, create_jira_issue, ...). |
| FR-AGB-03 | P0 | MCP server support: agents can call tools exposed via Model Context Protocol (MCP) servers registered at the org level. |
| FR-AGB-04 | P0 | Prompt editor with variable placeholders (`{{input.x}}`, `{{context.doc}}`) and linter. |
| FR-AGB-05 | P0 | Test harness: run against sample inputs; view traces (prompt, tokens, tool calls, latency, cost). |
| FR-AGB-06 | P0 | Eval harness: named test suites (input + expected assertions) with pass/fail and regression tracking. |
| FR-AGB-07 | P0 | Versioning: every publish creates an immutable version; rollback and diff supported. |
| FR-AGB-08 | P0 | Deployment targets: chat (usable in assistant), webhook URL, scheduled (cron via Cloud Scheduler), connector trigger (e.g., "new file in Drive folder"). |
| FR-AGB-09 | P0 | Guardrails: max tool-call count, max tokens, PII strategy, rate limit, timeout, allowed tools, allowed connectors. |
| FR-AGB-10 | P0 | Run history: every execution logged with inputs, outputs, trace, cost, user, trigger source. |
| FR-AGB-11 | P1 | Multi-step sub-agent orchestration (DAG or LangGraph-style state machine). |
| FR-AGB-12 | P1 | Human-in-the-loop approval steps (pause for reviewer, resume via UI). |
| FR-AGB-13 | P1 | Agent templates library: start-from templates (QA bot, report summarizer, invoice processor, meeting-notes summarizer). |
| FR-AGB-14 | P2 | A/B test two agent versions on a % of traffic. |

**Agent execution trace must capture (for every run):** timestamp, trigger, user, inputs, model(s), system prompt hash, each tool call (name, args, latency, status, error), each LLM call (provider, model, tokens in/out, latency, cost), final output, citations, errors, overall latency, overall cost.

### 6.14 Industry Templates

| ID | Priority | Requirement |
|---|---|---|
| FR-TMPL-01 | P0 | Seed templates: Food Manufacturing, Recruitment, Retail, Real Estate (per architecture blueprint §3). |
| FR-TMPL-02 | P0 | Each template bundles: document types + schemas + extraction prompts + validation rules + dashboard widgets + report templates + sample documents + starter agents. |
| FR-TMPL-03 | P0 | Template applied at onboarding inserts records into `document_types`, `custom_fields`, `dashboard_configs`, `report_templates`, `agents`. |
| FR-TMPL-04 | P0 | Tenants can customize (add/remove custom fields, edit prompts, add widgets). |
| FR-TMPL-05 | P1 | Admin Console to manage template catalog (add/update/version). |
| FR-TMPL-06 | P1 | "Update to latest template" migration tool with diff and opt-in. |

### 6.15 Admin Console (Customer-Facing)

| ID | Priority | Requirement |
|---|---|---|
| FR-ADM-01 | P0 | Org settings: name, logo, timezone, language, data region, default parser, PII strategy. |
| FR-ADM-02 | P0 | Users & teams: invite, role change, remove, MFA enforcement. |
| FR-ADM-03 | P0 | SSO & SCIM configuration. |
| FR-ADM-04 | P0 | Connectors: connect/disconnect, scope config, sync status, redaction rules. |
| FR-ADM-05 | P0 | Document types & schemas: editor for base + custom fields. |
| FR-ADM-06 | P0 | Agents: list, versions, usage, publish scope, disable. |
| FR-ADM-07 | P0 | Billing: plan, seats, usage, invoices, payment method. |
| FR-ADM-08 | P0 | Audit log viewer with filters (user, action, entity, date range, status). |
| FR-ADM-09 | P0 | Export / delete organization data (DSAR / tenant offboarding). |
| FR-ADM-10 | P1 | Feature flags per org (toggle beta features). |

### 6.16 Platform Superadmin Console (Biz2Bricks-Internal)

| ID | Priority | Requirement |
|---|---|---|
| FR-SADM-01 | P0 | List all orgs with filter (plan, status, created_at, last_active, usage). |
| FR-SADM-02 | P0 | Drill into an org: users, usage, connectors, agents, audit. |
| FR-SADM-03 | P0 | Adjust plan or quota overrides with justification (logged). |
| FR-SADM-04 | P0 | Impersonation with audit and reason. |
| FR-SADM-05 | P0 | Template catalog management (CRUD, versioning). |
| FR-SADM-06 | P0 | Model catalog (register/remove LLM models, set per-plan defaults, cost per token). |
| FR-SADM-07 | P0 | Feature flag management (global + per-org). |
| FR-SADM-08 | P0 | Incident tools: pause a tenant, force-rotate secrets, kill a bulk job. |
| FR-SADM-09 | P1 | System health dashboard (SLOs, queue depth, error rates). |

### 6.17 Observability (User-Facing)

| ID | Priority | Requirement |
|---|---|---|
| FR-OBS-01 | P0 | Per-query "Explain" view: route taken, SQL generated, sources cited, latency and cost. |
| FR-OBS-02 | P0 | Agent run detail: full trace (see FR-AGB-10). |
| FR-OBS-03 | P0 | Audit log searchable by entity and actor. |
| FR-OBS-04 | P1 | Cost per feature per month per org (customer-visible FinOps). |
| FR-OBS-05 | P1 | Export audit log to SIEM (S3, BigQuery, Splunk). |

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Dimension | Target |
|---|---|
| Chat first-token latency (P50) | ≤ 1.5 s |
| Chat first-token latency (P95) | ≤ 3.5 s |
| Search result (top-10) latency (P50) | ≤ 400 ms |
| Search result (top-10) latency (P95) | ≤ 1.0 s |
| Document parsing (single PDF ≤10 pages, Gemini) | ≤ 15 s P95 |
| Structured extraction (single doc, simple schema) | ≤ 8 s P95 |
| Bulk job throughput | ≥ 3 concurrent docs per job worker; horizontally scalable |
| API endpoint (non-LLM) P95 | ≤ 250 ms |
| Onboarding time-to-value (signup → first extraction visible) | ≤ 5 min |

### 7.2 Scale

- **Tenants**: 10,000 paying tenants by end of Year 1.
- **Peak concurrency**: 5,000 concurrent chat sessions.
- **Documents**: 1B documents across all tenants.
- **Embeddings**: 10B vectors (per-tenant stores).
- **Connector sync**: 500M documents/month incremental.

### 7.3 Availability

| SLO | Target |
|---|---|
| API uptime | 99.9% monthly |
| Chat/Search uptime | 99.9% monthly |
| Bulk processing | 99.5% (async, resumable) |
| Planned maintenance | ≤ 4 hours/month, announced 48h ahead |

### 7.4 Durability

- **Documents in GCS**: 11×9s durability (Google default).
- **PostgreSQL**: HA regional, point-in-time recovery 35 days, backups tested quarterly.
- **Zero data loss** on bulk job failure; jobs resume from checkpoint.

### 7.5 Security

- AES-256 at rest (GCS, Cloud SQL, Secret Manager).
- TLS 1.3 in transit.
- Per-tenant encryption keys via Cloud KMS for Enterprise (CMEK).
- PII detection and redaction at ingest (configurable).
- SAST / DAST / dependency scanning in CI.
- Annual external pen test.

### 7.6 Compliance

- **SOC 2 Type II**: controls designed from day 1, audit within 12 months of GA.
- **GDPR**: DSAR support (export + delete), data region pinning, EU DPA template.
- **HIPAA**: not claimed in v2; do not accept PHI (block list in ToS).
- **ISO 27001**: aligned controls, certification considered Year 2.

### 7.7 Internationalization

- UI strings externalized (i18n-ready); English only at GA.
- Dates/times localized by user timezone.
- Document parsing supports multilingual (per Gemini/LlamaParse).

### 7.8 Accessibility

- WCAG 2.1 AA baseline (contrast, keyboard navigation, ARIA).

### 7.9 Cost

- LLM cost per interaction: **< $0.03 average** for chat, **< $0.15 average** for extraction per document.
- Gross margin target ≥ 70% at scale.

### 7.10 Extensibility

- New connector: a new module implementing `Connector` protocol, no core code changes.
- New document type: a template entry, no deploy required per tenant.
- New agent tool: register in tool registry, exposed to Agent Builder automatically.
- New LLM model: register in model catalog, available per plan gate.

---

## 8. System Architecture (3-App SaaS)

### 8.1 High-Level Topology

```
                           ┌──────────────────────────────────┐
                           │         Users (Web)              │
                           └──────────────┬───────────────────┘
                                          │ HTTPS
                           ┌──────────────▼───────────────────┐
                           │  CloudFlare / Google Cloud CDN   │
                           │  WAF + DDoS                      │
                           └──────────────┬───────────────────┘
                                          │
                           ┌──────────────▼───────────────────┐
                           │  Cloud Load Balancer             │
                           └──────┬─────────────────────┬─────┘
                                  │                     │
                 ┌────────────────▼──┐   ┌──────────────▼────────────────┐
                 │  Frontend (Next.js)│   │   Public API endpoints        │
                 │  Cloud Run         │   │   (webhooks, OAuth callbacks) │
                 │                    │   │                                │
                 │  - UI              │   │                                │
                 │  - Route handlers  │   │                                │
                 │    act as BFF      │   │                                │
                 └────────┬───────────┘   └─────────────┬──────────────────┘
                          │ internal (VPC)               │
          ┌───────────────┴────────────┬────────────────┴────────────────┐
          │                            │                                  │
┌─────────▼────────────┐   ┌───────────▼────────────┐       ┌────────────▼────────────┐
│  Admin Backend       │   │  AI Backend (FastAPI)  │       │  Worker services        │
│  (FastAPI)           │   │                        │       │                          │
│                      │   │  - Agents              │       │  - Connector sync        │
│  - Identity          │   │  - Hybrid Router       │       │  - Bulk processor        │
│  - RBAC              │   │  - Text-to-SQL         │       │  - Embedding generator   │
│  - Org mgmt          │   │  - RAG / FS query      │       │  - Report scheduler      │
│  - Billing           │   │  - Chat streaming      │       │  - Agent executor        │
│  - Audit             │   │  - Agent runtime       │       │                          │
│  - Connectors meta   │   │                        │       │  (Cloud Run jobs /       │
│  - Platform admin    │   │                        │       │   Pub/Sub consumers)     │
└──────┬───────────────┘   └──────────┬─────────────┘       └────────────┬─────────────┘
       │                              │                                   │
       └──────────────────────────────┴───────────────────┬───────────────┘
                                                          │
             ┌────────────────────────────────────────────┴─────────────────┐
             │                       Data Plane                             │
             │                                                              │
             │  PostgreSQL 16 (Cloud SQL, HA, RLS)                          │
             │  GCS (per-tenant prefixes)                                   │
             │  Gemini File Search (per-tenant stores)                      │
             │  Memorystore Redis (cache, rate limit, sessions)             │
             │  Pub/Sub (events)                                            │
             │  Cloud Tasks (scheduling)                                    │
             │  Secret Manager                                              │
             │  BigQuery (analytics, usage rollups)                         │
             │  OpenTelemetry Collector → Cloud Trace / Logs                │
             └──────────────────────────────────────────────────────────────┘
```

### 8.2 Responsibility Split — Admin Backend vs. AI Backend

| Capability | Admin Backend | AI Backend |
|---|---|---|
| Sign up, login, MFA | ✓ | — |
| SSO / SCIM | ✓ | — |
| Org / user / team CRUD | ✓ | — |
| Billing (Stripe) | ✓ | — |
| Quota checks | ✓ (source of truth) | Reads cached quota |
| Audit log writes | ✓ (writes + stores) | Emits events → Admin consumes |
| Connectors: OAuth + config | ✓ | — |
| Connectors: sync workers | — | ✓ (worker) |
| Agents: metadata / versioning | ✓ | Consumes at runtime |
| Agents: execution runtime | — | ✓ |
| Document parsing | — | ✓ |
| Extraction | — | ✓ |
| Content generation | — | ✓ |
| Chat / search queries | — | ✓ |
| BI reports | — | ✓ |
| Dashboard data | — | ✓ (reads from DB) |

The **Admin Backend is the system of record for metadata** (users, orgs, quotas, agents, connectors, audit). The **AI Backend is the runtime** for intelligence operations. They communicate via:

1. A **shared PostgreSQL** (with strict schema ownership per domain; see §9) — both services read; Admin writes metadata, AI writes runtime data (jobs, generations, usage).
2. **Pub/Sub events** for asynchronous cross-service signals (`org.created`, `quota.exceeded`, `agent.published`, `document.processed`).
3. **Internal REST APIs** with service-to-service mTLS for synchronous calls (`AdminClient.check_quota()`, `AdminClient.get_org_context()`).

### 8.3 Tech Stack (per app)

**AI Backend (`biz2bricks-ai-v2`)**
- Python 3.12 on Cloud Run (min 1, max 50 instances; resource: 4 vCPU / 8 GiB).
- FastAPI + uvicorn.
- LangGraph 1.0+, LangChain 1.2+.
- LLMs: OpenAI (gpt-5.1-codex-mini, gpt-5-mini, gpt-4o-mini), Gemini (gemini-3-flash, gemini-3-pro), Anthropic (claude-sonnet-4, claude-haiku-4.5) — swappable.
- DuckDB for in-memory SQL on sheets.
- LlamaParse for fallback document parsing.
- Gemini File Search for RAG.
- SQLAlchemy 2.0 async + asyncpg.
- Redis (Memorystore) for session, rate-limit, cache.
- Pub/Sub for async work; Cloud Tasks for scheduled jobs.
- OpenTelemetry + Cloud Trace.

**Admin Backend (`biz2bricks-admin-v2`)**
- Python 3.12 on Cloud Run (min 2, max 20 instances).
- FastAPI + uvicorn.
- SQLAlchemy 2.0 async.
- WorkOS SDK or Auth0 SDK (SSO/SCIM).
- Stripe SDK.
- OpenTelemetry.

**Frontend (`biz2bricks-fe-v2`)**
- Next.js 15 App Router + React 19.
- TypeScript (strict).
- Tailwind CSS + shadcn/ui + Framer Motion.
- TanStack Query for server state; Zustand for UI state.
- NextAuth.js (with WorkOS/Auth0 providers).
- SSE via native `fetch` + `ReadableStream`.
- Vercel AI SDK or custom streaming for LLM UX.

**Shared libraries**
- `biz2bricks-core` (Python): ORM models, dataclasses, shared utilities. Already exists in v1; continue pattern.
- `biz2bricks-ts-types` (TypeScript): shared types generated from OpenAPI schemas of both backends.

### 8.4 Repository Layout (Monorepo or Polyrepo)

Recommendation: **polyrepo** (three independent repos + one shared types repo). Rationale: independent deploy cadence, clearer ownership, and Claude Code / Codex workflows work best with focused repos.

```
biz2bricks-ai-v2/           # AI Backend (Python)
biz2bricks-admin-v2/        # Admin Backend (Python)
biz2bricks-fe-v2/           # Frontend (TypeScript/Next.js)
biz2bricks-core/            # Shared Python lib (ORM models, utils)
biz2bricks-ts-types/        # Shared TS types (generated from OpenAPI)
biz2bricks-infra/           # Terraform + GitHub Actions templates
```

### 8.5 Networking and Security

- All Cloud Run services run inside a shared **VPC** with a **VPC connector**.
- Cloud SQL is **private-IP only**; no public access.
- Admin ↔ AI backend calls go **in-VPC** with **mTLS** (certificates from Google-managed CA or HashiCorp Vault).
- Only the Frontend and explicit public-API services are exposed to the internet, behind **Cloud Armor** (WAF rules, rate limits, bot blocking).
- **Secret Manager** holds every credential; rotated quarterly.
- All inbound traffic goes through **Cloud CDN** with DDoS protection.

### 8.6 Multi-Tenancy Isolation

Defense in depth, in this order:

1. **JWT scope**: every token contains `org_id` and `role`.
2. **Application middleware**: validates `X-Organization-ID` matches `org_id` in JWT; injects `tenant_context` into request state.
3. **PostgreSQL RLS**: every tenant-scoped table has an RLS policy keyed on `current_setting('app.current_org_id')`. The application sets this at connection check-out.
4. **GCS**: per-org prefix (`gs://{bucket}/orgs/{org_id}/...`) with IAM conditions.
5. **Gemini File Search**: one store per tenant per document category (`tenant-{org_id}-contracts`, etc.).
6. **Redis**: namespaced keys (`{org_id}:{feature}:{key}`).

---

## 9. Data Model

### 9.1 PostgreSQL Schemas (Logical)

Organize into schemas per domain (`public` can host shared):

- `identity` (users, organizations, teams, memberships, sso_configs, invitations)
- `billing` (subscription_tiers, organization_subscriptions, invoices, usage_records, token_usage, resource_usage)
- `content` (folders, documents, document_folders, document_types, custom_fields)
- `processing` (processing_jobs, bulk_jobs, bulk_job_documents, document_generations, extraction_records, extraction_schemas, reports)
- `search` (file_search_stores, rag_query_cache, connector_sources, connector_scopes, connector_sync_jobs, indexed_documents, acl_cache, embeddings_metadata)
- `agents` (agents, agent_versions, agent_tools, agent_runs, agent_schedules, agent_evals, tool_registry)
- `memory` (conversation_summaries, memory_entries, user_preferences, chat_sessions, messages)
- `audit` (audit_log, impersonation_log, system_events)
- `admin` (feature_flags, template_catalog, model_catalog, platform_users)

**Row-Level Security** is enabled on every table in `content`, `processing`, `search`, `agents`, `memory`, and `audit` using `org_id`.

### 9.2 Key Tables (new / changed for v2)

```sql
-- identity
CREATE TABLE identity.organizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  industry_id TEXT NOT NULL,
  data_region TEXT NOT NULL DEFAULT 'us',
  company_size TEXT,
  timezone TEXT DEFAULT 'UTC',
  default_language TEXT DEFAULT 'en',
  logo_url TEXT,
  settings JSONB DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'active',    -- active|suspended|deleted
  created_at TIMESTAMPTZ DEFAULT now(),
  deleted_at TIMESTAMPTZ
);

CREATE TABLE identity.users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email CITEXT UNIQUE NOT NULL,
  email_verified_at TIMESTAMPTZ,
  password_hash TEXT,
  mfa_enabled BOOLEAN DEFAULT FALSE,
  totp_secret TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE identity.memberships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES identity.users(id) ON DELETE CASCADE,
  org_id UUID REFERENCES identity.organizations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('owner','admin','member','viewer','auditor')),
  invited_by UUID REFERENCES identity.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (user_id, org_id)
);

CREATE TABLE identity.sso_configs (
  org_id UUID PRIMARY KEY REFERENCES identity.organizations(id),
  protocol TEXT CHECK (protocol IN ('saml','oidc')),
  provider TEXT,                             -- okta|azure_ad|google|generic
  metadata JSONB,
  scim_token_hash TEXT,
  enforced BOOLEAN DEFAULT FALSE
);

-- search
CREATE TABLE search.connector_sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES identity.organizations(id),
  connector_id TEXT NOT NULL,                -- google_drive, slack, ...
  display_name TEXT,
  auth_secret_ref TEXT,                      -- Secret Manager path
  status TEXT DEFAULT 'active',
  last_full_sync_at TIMESTAMPTZ,
  last_delta_cursor TEXT,
  created_by UUID REFERENCES identity.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (org_id, connector_id)
);

CREATE TABLE search.connector_scopes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id UUID NOT NULL REFERENCES search.connector_sources(id) ON DELETE CASCADE,
  org_id UUID NOT NULL,
  external_id TEXT NOT NULL,                 -- folder/channel id in source
  display_name TEXT,
  scope_type TEXT,                           -- folder|channel|space|drive
  settings JSONB DEFAULT '{}'
);

CREATE TABLE search.indexed_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL,
  source_id UUID REFERENCES search.connector_sources(id),
  external_id TEXT,
  source_path TEXT,                          -- full path/URL in source
  title TEXT,
  mime_type TEXT,
  modified_at TIMESTAMPTZ,
  indexed_at TIMESTAMPTZ,
  acl JSONB,                                 -- [{type:'user',email:...}, ...]
  metadata JSONB,
  content_hash TEXT,
  file_search_store_id TEXT,
  status TEXT,                               -- indexed|pending|error|redacted
  UNIQUE (org_id, source_id, external_id)
);

-- agents
CREATE TABLE agents.agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  avatar_url TEXT,
  current_version_id UUID,
  publish_scope TEXT DEFAULT 'draft',        -- draft|private|org|marketplace_private|marketplace_public
  created_by UUID,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (org_id, name)
);

CREATE TABLE agents.agent_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
  version_number INT NOT NULL,
  input_schema JSONB,
  output_schema JSONB,
  system_prompt TEXT,
  model TEXT,
  tools JSONB,                               -- list of tool refs + config
  knowledge_sources JSONB,                   -- folders, stores, connectors
  guardrails JSONB,                          -- max_tokens, timeout, pii, ...
  created_by UUID,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (agent_id, version_number)
);

CREATE TABLE agents.agent_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL,
  agent_id UUID NOT NULL,
  agent_version_id UUID,
  trigger TEXT,                              -- chat|webhook|schedule|connector
  input JSONB,
  output JSONB,
  trace JSONB,                               -- full trace of tool+LLM calls
  status TEXT,                               -- running|success|error|timeout
  cost_usd NUMERIC(10,4),
  latency_ms INT,
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  user_id UUID
);

CREATE TABLE agents.tool_registry (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tool_id TEXT UNIQUE NOT NULL,              -- global id, e.g. "search.file_search"
  display_name TEXT,
  description TEXT,
  input_schema JSONB,
  output_schema JSONB,
  implementation TEXT,                       -- "builtin" | "mcp:{server_id}"
  required_scopes JSONB,                     -- connectors/permissions needed
  is_enabled BOOLEAN DEFAULT TRUE
);

-- audit
CREATE TABLE audit.audit_log (
  id BIGSERIAL PRIMARY KEY,
  org_id UUID,
  user_id UUID,
  actor_type TEXT,                           -- user|service|platform
  action TEXT NOT NULL,                      -- e.g., document.uploaded
  entity_type TEXT,
  entity_id TEXT,
  metadata JSONB,
  ip_address INET,
  user_agent TEXT,
  status TEXT,                               -- success|failure
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON audit.audit_log (org_id, created_at DESC);
CREATE INDEX ON audit.audit_log (action, created_at DESC);
```

### 9.3 Carry-Forward From v1

The following v1 models stay mostly as-is (migrate into `processing`, `content`, `billing`, `memory` schemas as appropriate). See `src/db/models.py` in `doc_intelligence_ai_v3.0`:

- `ProcessingJobModel`, `BulkJobModel`, `BulkJobDocumentModel` → `processing.*`
- `DocumentGenerationModel` → `processing.document_generations`
- `DocumentModel`, `FolderModel`, `DocumentFolderModel` → `content.*`
- `SubscriptionTierModel`, `OrganizationSubscriptionModel`, `TokenUsageRecordModel`, `ResourceUsageRecordModel`, `UsageAggregationModel` → `billing.*`
- `ConversationSummaryModel`, `MemoryEntryModel`, `UserPreferenceModel` → `memory.*`
- `RAGQueryCacheModel` → `search.rag_query_cache`
- `FileSearchStoreModel` → `search.file_search_stores`

### 9.4 Storage (GCS) Path Conventions

```
gs://{env}-biz2bricks-documents/
  orgs/
    {org_id}/
      uploads/         # raw uploaded files
      parsed/          # Markdown + metadata from parser
      generated/       # summaries, FAQs, questions
      extracted/       # CSV/XLSX/JSON of extracted records
      reports/         # generated BI reports
      connectors/
        {connector_id}/{external_id}/...  # cached content from connectors
```

Per-org KMS key (Enterprise).

### 9.5 Vector Store Layout

Per tenant, a Gemini File Search store is created **per knowledge category**:

- `{org_id}:uploads` (everything uploaded)
- `{org_id}:connector:{connector_id}` (one per connected source)
- `{org_id}:templates:{doc_type_key}` (e.g., `{org_id}:templates:invoice`)

The Hybrid Query Router selects which stores to query based on the question and filters.

### 9.6 Knowledge Graph (v2 lightweight)

Represented inside PostgreSQL for v2 (Neo4j deferred):

```sql
CREATE TABLE search.kg_entities (
  id UUID PRIMARY KEY,
  org_id UUID,
  type TEXT,                      -- person|document|project|account|topic
  canonical_name TEXT,
  attributes JSONB,
  created_at TIMESTAMPTZ
);
CREATE TABLE search.kg_edges (
  id UUID PRIMARY KEY,
  org_id UUID,
  from_entity UUID,
  to_entity UUID,
  relation TEXT,                  -- mentioned_in|owner_of|replied_to|...
  weight NUMERIC,
  source_event_id UUID
);
```

Entities are resolved from connector metadata (Slack users, Drive owners, Jira assignees) and from document extraction (vendors, candidates, properties).

---

## 10. API Contracts (Summary)

All APIs use:
- OpenAPI 3.1 spec (source of truth; published at `/openapi.json` on each service).
- Resource-oriented, JSON bodies, snake_case fields.
- Standard headers: `Authorization: Bearer <JWT>`, `X-Organization-ID`, `X-Request-ID`.
- Standard error envelope: `{ "error": { "code": "...", "message": "...", "details": {...}, "trace_id": "..." } }`.
- Pagination: `limit` (default 20, max 100) + `cursor`.

### 10.1 Admin Backend (selected endpoints)

```
POST   /auth/signup
POST   /auth/login
POST   /auth/logout
POST   /auth/mfa/setup
POST   /auth/mfa/verify
POST   /auth/refresh
GET    /auth/me

POST   /orgs
GET    /orgs
GET    /orgs/{org_id}
PATCH  /orgs/{org_id}
DELETE /orgs/{org_id}
POST   /orgs/{org_id}/transfer-owner

POST   /orgs/{org_id}/invitations
GET    /orgs/{org_id}/members
PATCH  /orgs/{org_id}/members/{user_id}
DELETE /orgs/{org_id}/members/{user_id}

POST   /orgs/{org_id}/teams
GET    /orgs/{org_id}/teams

POST   /orgs/{org_id}/sso
GET    /orgs/{org_id}/sso
POST   /orgs/{org_id}/scim/token

POST   /orgs/{org_id}/connectors
GET    /orgs/{org_id}/connectors
POST   /orgs/{org_id}/connectors/{connector_id}/oauth/start
GET    /orgs/{org_id}/connectors/{connector_id}/oauth/callback
POST   /orgs/{org_id}/connectors/{source_id}/scopes
POST   /orgs/{org_id}/connectors/{source_id}/sync      # trigger manual sync
DELETE /orgs/{org_id}/connectors/{source_id}

GET    /orgs/{org_id}/agents
POST   /orgs/{org_id}/agents
GET    /orgs/{org_id}/agents/{agent_id}
PATCH  /orgs/{org_id}/agents/{agent_id}
POST   /orgs/{org_id}/agents/{agent_id}/versions
POST   /orgs/{org_id}/agents/{agent_id}/publish
GET    /orgs/{org_id}/agents/{agent_id}/runs

GET    /orgs/{org_id}/billing/subscription
POST   /orgs/{org_id}/billing/subscription              # change plan
GET    /orgs/{org_id}/billing/usage
GET    /orgs/{org_id}/billing/invoices
GET    /orgs/{org_id}/billing/invoices/{invoice_id}/pdf

GET    /orgs/{org_id}/audit
GET    /orgs/{org_id}/audit/export

# Platform admin
GET    /platform/orgs
GET    /platform/orgs/{org_id}
POST   /platform/orgs/{org_id}/impersonate
POST   /platform/orgs/{org_id}/quota-override
GET    /platform/templates
POST   /platform/templates
GET    /platform/models
POST   /platform/feature-flags
```

### 10.2 AI Backend (selected endpoints — evolves v1 surface)

```
POST   /api/v2/chat                       # SSE streaming, hybrid router
POST   /api/v2/search                     # unified search (docs + connectors)
POST   /api/v2/answer                     # generative answer variant of search

POST   /api/v2/documents/parse
POST   /api/v2/documents/summarize
POST   /api/v2/documents/faqs
POST   /api/v2/documents/questions
POST   /api/v2/documents/generate-all

POST   /api/v2/extraction/analyze-fields
POST   /api/v2/extraction/schemas
GET    /api/v2/extraction/schemas
POST   /api/v2/extraction/extract
GET    /api/v2/extraction/records
POST   /api/v2/extraction/export

POST   /api/v2/sheets/analyze
POST   /api/v2/sheets/preview

POST   /api/v2/bulk/upload
POST   /api/v2/bulk/jobs
GET    /api/v2/bulk/jobs/{job_id}
POST   /api/v2/bulk/jobs/{job_id}/cancel
POST   /api/v2/bulk/jobs/{doc_id}/retry

POST   /api/v2/reports                     # create
GET    /api/v2/reports/{report_id}
GET    /api/v2/reports/{report_id}/download
GET    /api/v2/dashboards/{dashboard_id}/data

POST   /api/v2/agents/{agent_id}:invoke    # synchronous
POST   /api/v2/agents/{agent_id}:run       # async run (returns run_id)
GET    /api/v2/agents/runs/{run_id}
GET    /api/v2/agents/runs/{run_id}/trace

POST   /api/v2/rag/stores                  # legacy compat
POST   /api/v2/rag/search
GET    /api/v2/sessions/{session_id}
DELETE /api/v2/sessions/{session_id}

# Internal / webhook
POST   /internal/events/document-uploaded
POST   /internal/events/connector-sync
POST   /internal/agents/schedule-tick
```

### 10.3 Streaming Protocol

Chat and agent runs stream via **SSE**. Event types:

```
event: status     data: {"phase":"routing"}
event: status     data: {"phase":"querying_sql"}
event: token      data: {"text":"Your total "}
event: citation   data: {"doc_id":"...", "source":"google_drive:...", "snippet":"..."}
event: tool_call  data: {"tool":"search", "args":{...}, "status":"started"}
event: tool_call  data: {"tool":"search", "status":"completed", "latency_ms":234}
event: sql        data: {"query":"SELECT ...","rows":[...]}
event: done       data: {"total_tokens":1234,"cost_usd":0.0125,"latency_ms":1800}
event: error      data: {"code":"...","message":"..."}
```

---

## 11. Event-Driven Architecture

All cross-service signals go through **Pub/Sub**. Event names follow `<domain>.<entity>.<action>` dot-convention.

| Topic | Producer | Consumer(s) | Payload |
|---|---|---|---|
| `org.organization.created` | Admin | AI (provision GCS prefixes, FS stores, templates), Search | `{org_id, industry_id, data_region, ...}` |
| `org.organization.deleted` | Admin | AI (cleanup), Search | `{org_id}` |
| `org.user.invited` | Admin | Notifier | `{org_id, user_id, email}` |
| `billing.quota.exceeded` | Admin | AI (block feature calls) | `{org_id, feature, limit, used}` |
| `billing.subscription.updated` | Admin | AI | `{org_id, plan, effective_at}` |
| `content.document.uploaded` | AI | Search, Agent Scheduler | `{org_id, doc_id, gcs_path, mime_type}` |
| `content.document.parsed` | AI | Search (index), AI (extract if auto) | `{org_id, doc_id, parsed_path}` |
| `content.document.extracted` | AI | Search, BI, Knowledge Graph | `{org_id, doc_id, schema_id, record_id}` |
| `search.connector.sync.completed` | AI | Knowledge Graph, Notifier | `{org_id, source_id, docs_indexed}` |
| `search.connector.doc.indexed` | AI | Knowledge Graph | `{org_id, source_id, external_id}` |
| `agent.run.started` | AI | Audit | `{run_id, agent_id, user_id}` |
| `agent.run.completed` | AI | Audit, Billing | `{run_id, status, cost, tokens}` |
| `agent.scheduled.tick` | Cloud Scheduler | AI | `{agent_id, schedule_id}` |
| `chat.message.sent` | AI | Audit | `{org_id, session_id, user_id, tokens}` |

Events are **at-least-once**, consumers must be **idempotent** (dedupe via `event_id`).

---

## 12. Security, Privacy, and Compliance

### 12.1 Authentication

- JWT (RS256), short-lived access token, refresh token rotation.
- Refresh tokens stored hashed; revocable.
- MFA available for all users; enforceable org-wide.
- SSO (SAML/OIDC) via WorkOS or Auth0.
- Service-to-service: mTLS between Admin and AI backends.

### 12.2 Authorization

- Coarse-grained: role (see §3.3).
- Fine-grained: resource ACLs (folder, agent, saved query).
- Connector ACL mirroring (the source's native permissions).

### 12.3 PII & Secrets

- PII detector runs on ingest (Microsoft Presidio or equivalent), configurable strategy per org (redact | mask | hash | block | off).
- Same detector available as a middleware for inbound chat input and outbound LLM output.
- API keys / connector credentials stored in Secret Manager, never in DB; only references (`secret://{path}`).

### 12.4 Data Residency

- `data_region` at org creation. Cloud SQL and GCS pinned to that region.
- LLM calls routed to same-region providers where possible (Vertex AI regions; OpenAI/Anthropic regional options).

### 12.5 Encryption

- AES-256 at rest (default GCP).
- CMEK via Cloud KMS for Enterprise tenants (one key per tenant).
- TLS 1.3 in transit.
- Application-layer encryption for especially sensitive fields (e.g., connector OAuth refresh tokens).

### 12.6 Audit

- Every state-changing action emits an `audit_log` row.
- `trace_id` (OTel) links audit entries to traces and logs.
- Retention: 1 year in hot storage, 7 years in BigQuery archive for Enterprise.

### 12.7 Data Subject Rights (GDPR)

- **Export** user data for a given user within an org (JSON archive + document ZIP).
- **Delete**: soft-delete user; cascade-delete PII; 30-day purge.
- **Portability** of full org data (documents + extracted records + agents).

### 12.8 Responsible AI

- Input PII stripping before prompts (configurable).
- Output moderation (profanity, hate speech filter via Vertex Safety or Claude's built-in guardrails).
- Model catalog labels models with licensing, data-training-opt-out status, data residency.
- **Bring-Your-Own-Key** (BYOK) for Enterprise: tenant provides their own OpenAI/Anthropic key, Biz2Bricks does not log prompts/outputs through our shared analytics for those tenants.

### 12.9 Abuse & Safety

- Rate limits per user + per org.
- Anomaly detection (sudden spike in tokens / agent runs / export volume) → alert.
- ToS blocklist for prompts attempting jailbreak / illegal content.

### 12.10 Compliance Program

- SOC 2 Type II: evidence collection via Vanta/Drata; audit within 12 months.
- GDPR: DPA available; sub-processor list published.
- HIPAA: excluded in v2; contract language prohibits uploading PHI.
- DSAR SLA: 30 days per GDPR.

---

## 13. Observability, SLOs, and Reliability

### 13.1 Telemetry

- **Traces**: OpenTelemetry SDK, exported to Cloud Trace. One trace per HTTP request; LLM calls and tool calls as spans. Trace IDs echoed in response headers (`X-Trace-ID`).
- **Metrics**: OTel metrics → Cloud Monitoring. Standard dimensions: `service`, `env`, `org_id` (bucketed into high-cardinality allowed set), `feature`.
- **Logs**: structured JSON logs, shipped to Cloud Logging; 30-day hot, 1-year archive in BigQuery.
- **LLM-specific**: every LLM call logs `provider`, `model`, `prompt_hash`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `cache_hit`.

### 13.2 SLOs

| Service | SLI | SLO |
|---|---|---|
| Frontend | Availability | 99.9% |
| Admin API | Availability | 99.95% |
| AI API (chat) | Availability | 99.9% |
| AI API (chat) | P95 first-token latency | ≤ 3.5 s |
| AI API (search) | P95 latency | ≤ 1.0 s |
| Bulk processing | Job success rate | ≥ 99% |
| Connector sync | Per-job success rate | ≥ 98% |

Error budget policy: if monthly burn exceeds 50% halfway through the month, feature freeze for the affected service until the team remediates.

### 13.3 Alerting

- Pager triggers: SLO burn, DB replication lag > 10s, Redis unreachable, LLM provider error rate > 5%, quota-check latency > 200ms.
- Slack-only: batch job failure > threshold, connector sync errors, anomaly flags.

### 13.4 LLM Evals

An offline eval harness (`biz2bricks-ai-v2/evals/`):

- Test suites per agent and per route (`chat`, `search`, `extraction`).
- Golden datasets per industry template.
- Metrics: accuracy (against labeled goldens), groundedness (claim-citation check), latency, cost.
- Runs on every PR that touches agents/prompts.
- Reports attached to PR; regression blocks merge.

### 13.5 Feature Flags

- `launchdarkly`/`unleash`/in-house via Cloud Storage-backed config.
- Per-org and per-user targeting.
- Rollouts gated behind flags for all new features.

---

## 14. DevOps, Deployment, and Environments

### 14.1 Environments

| Env | Purpose | Data | Traffic |
|---|---|---|---|
| `dev` | Engineer integration | Synthetic + mock connectors | Zero prod |
| `staging` | Pre-prod smoke + perf | Mirror-structure prod, anonymized | Internal users |
| `prod` | Customers | Real | All |
| `sandbox` | Sales/demo | Ephemeral per SE | Demos |

### 14.2 Infrastructure as Code

- **Terraform** for all GCP resources, in `biz2bricks-infra`.
- Per-env workspaces: `dev`, `staging`, `prod`, plus regional modules (us, eu, apac).
- Modules: `cloud-run-service`, `cloudsql-instance`, `gcs-bucket-tenant`, `pubsub-topic`, `secret`, `cloud-scheduler-job`.

### 14.3 CI/CD

- **GitHub Actions** per repo:
  - PR: lint, type-check (mypy/ts), unit tests, SAST (Bandit/Semgrep), dependency scan (Snyk).
  - Merge to `main`: build image, push to Artifact Registry, deploy to `dev`.
  - Tag `v*.*.*`: deploy to `staging` → manual gate → `prod`.
  - Blue/green on Cloud Run via revisions + traffic split; automatic rollback on SLO burn.

### 14.4 Database Migrations

- **Alembic** for Python backends.
- Expand-contract pattern for schema changes.
- Zero-downtime: add columns nullable; deploy app writing new schema; backfill; flip reads; drop old.

### 14.5 Secret Management

- Secret Manager per env.
- Rotation: 90 days automatic for DB passwords; 180 days for third-party API keys; manual for signing keys (annual).

### 14.6 Backup & DR

- Cloud SQL: automated backups every 6h, PITR 35 days.
- GCS: versioning enabled on content buckets.
- DR: warm standby in secondary region; RTO 4h, RPO 15 min.
- Quarterly DR drills.

---

## 15. Release Phasing and Roadmap

### 15.1 Phasing Strategy

v2 ships in three milestones. Each adds shippable value; none requires the next to be launched.

#### Milestone M1 — "SaaS Foundation" (target: 12 weeks)

**Goal:** v1 capabilities repackaged as production SaaS with multi-tenancy, billing, and admin.

Scope:
- Admin Backend skeleton: identity (email + Google OAuth), orgs, teams, memberships, RBAC.
- Stripe integration + plans + quota checks.
- v1 AI Backend (`doc_intelligence_ai_v3.0`) hardened and re-pointed to Admin for identity/quota.
- Frontend: login, onboarding, dashboard, documents, chat, extraction, reports — wired to both backends.
- Audit log and basic observability.
- **Single region (US)**.
- Industry templates: Food Manufacturing and Recruitment (two, not four).
- No connectors yet.
- No Agent Builder UI yet (but infra scaffold).

Exit criteria:
- 5 beta customers onboarded self-serve.
- 99.9% uptime over last 30 days.
- SOC 2 Type I audit started.

#### Milestone M2 — "Unified Search" (target: 10 weeks)

**Goal:** Enterprise-search surface with first 5 connectors.

Scope:
- Connector framework + module contract.
- Connectors: Google Drive, Microsoft OneDrive/SharePoint, Slack, Gmail, Confluence.
- Connector sync workers (full + delta + webhook-driven).
- ACL mirroring and permission-aware search.
- Unified search UI (facets, filters, previews, answer toggle).
- Hybrid Query Router v1 (SQL + RAG + connector).
- Text-to-SQL engine.
- Knowledge graph v1 (people/docs/topics).
- SSO (SAML/OIDC), SCIM.
- Second region (EU).

Exit criteria:
- 25 paying tenants.
- Median search latency P50 ≤ 400 ms.
- ≥ 2 industry templates added (Retail + Real Estate).

#### Milestone M3 — "Agent Builder GA" (target: 12 weeks)

**Goal:** Agents as a first-class surface; Fast-follow connectors.

Scope:
- Agent Builder UI (full authoring, test, eval, version, publish).
- Tool registry + MCP server support.
- Scheduled + webhook + connector-triggered agents.
- Run traces and eval harness.
- Fast-follow connectors: Jira, Notion, Dropbox/Box, Salesforce, HubSpot.
- Human-in-the-loop step support.
- BYOK (Enterprise) for OpenAI/Anthropic/Gemini keys.
- SOC 2 Type II audit window begins.

Exit criteria:
- 100 paying tenants.
- 1,000 published agents across the platform.
- NRR ≥ 110%.

### 15.2 Roadmap Beyond v2

- **v2.x minor releases** for connector additions, template additions, model updates.
- **v3 themes**: public agent marketplace, dedicated single-tenant deployments, advanced workflows (DAGs with approvals and rollback), native mobile app, HIPAA SKU.

### 15.3 Sequenced Backlog (Sample Engineering Epics)

1. Extract identity out of v1 AI backend into Admin Backend.
2. Introduce RLS everywhere; regression tests for cross-tenant leakage.
3. Stripe integration + usage metering finalization.
4. Connector framework (`Connector` protocol + OAuth + scheduler + delta).
5. Connector #1–#5 implementations.
6. Hybrid Query Router + Text-to-SQL engine.
7. Unified Search UI in Next.js.
8. SSO/SCIM via WorkOS.
9. Agent Builder UI (tabs: Design / Test / Eval / Deploy).
10. Tool registry + MCP integration.
11. Agent runtime hardening (retries, traces, cost caps).
12. Knowledge graph extractors + storage.
13. Observability: OTel everywhere, LLM eval harness, per-tenant cost dashboards.
14. Platform Superadmin console.
15. DR drills, secret rotation automation, pen test remediation.

---

## 16. Success Metrics and Acceptance Criteria

### 16.1 Business Metrics

| Metric | Target (End of Year 1) |
|---|---|
| Paying tenants | 1,000 |
| Monthly recurring revenue (MRR) | $250,000 |
| LTV:CAC | ≥ 10:1 |
| Monthly churn (logo) | ≤ 3% |
| Net revenue retention | ≥ 110% |
| Weekly Active Agents per tenant (WAA/T) | ≥ 5 |
| Time-to-value (onboarding → first extraction) | ≤ 5 min (P50) |

### 16.2 Product Metrics

| Area | Metric | Target |
|---|---|---|
| Activation | % of signed-up orgs that extract ≥1 document in week 1 | ≥ 70% |
| Activation | % of orgs that connect ≥1 connector within 14 days | ≥ 50% |
| Engagement | Weekly active users per tenant (WAU/T) | ≥ 3 |
| Search | Zero-result query rate | ≤ 5% |
| Search | Answer-clicked-through rate | ≥ 60% |
| Agents | % of tenants with ≥1 published agent | ≥ 40% |
| Extraction | Field-level accuracy (vs. human-labeled goldens) | ≥ 92% |
| Reliability | 30-day uptime (chat + search) | ≥ 99.9% |

### 16.3 Engineering Acceptance Criteria per Milestone

Each milestone has a **go/no-go checklist**:

- Functional tests: 100% of P0 FRs automated and passing in CI.
- Performance: load tests at 2× projected peak, all NFR latency targets met P95.
- Security: SAST/DAST clean on P0+P1; pen-test findings ≤ medium severity.
- Observability: every P0 endpoint has traces, logs, metrics, and an SLO alert.
- Documentation: API docs (OpenAPI), runbooks for every on-call scenario, a one-pager per major feature.
- Data migrations: rehearsed on staging with production-scale data.

### 16.4 Definition of Done (per feature)

A feature is "done" when:
1. FRs mapped to test IDs, all passing in CI.
2. Feature flagged and toggled on in `dev`/`staging`.
3. Observability in place (trace, metric, log fields, dashboard row, alert if applicable).
4. Security review signed off (PII handling, RBAC, RLS if new table).
5. Docs: user-facing help article + internal runbook + API docs updated.
6. Rollout plan: canary at 5% → 25% → 100% with SLO-based auto-rollback.

---

## 17. Open Questions, Assumptions, and Risks

### 17.1 Open Questions

| # | Question | Needs answer by |
|---|---|---|
| OQ-01 | WorkOS vs. Auth0 for SSO/SCIM? (Pricing, UX, SDK maturity.) | End of M1 planning |
| OQ-02 | Single multi-region PostgreSQL vs. per-region clusters? | Before M2 region rollout |
| OQ-03 | Neo4j vs. PostgreSQL for knowledge graph? | Before M3 advanced KG features |
| OQ-04 | MCP server hosting: in-Biz2Bricks runtime vs. customer-hosted only? | M3 |
| OQ-05 | Is the Sheets agent dissolved into Document+Extraction, or kept separate? | M1 design freeze |
| OQ-06 | Default LLM provider for chat (Gemini vs. GPT vs. Claude)? What's the routing heuristic? | M1 |
| OQ-07 | Offline mode / on-prem interest among target SMBs? (Likely deferred.) | Post-GA |
| OQ-08 | Stripe alone or Stripe+Metronome for metered billing? | M1 |
| OQ-09 | Hosted vector search (Pinecone / Turbopuffer) vs. Gemini File Search as the primary? | M2 |
| OQ-10 | Legal: acceptance of multi-tenant shared embeddings (no bleed, but customers may still object)? | M2 |

### 17.2 Assumptions

- GCP remains our primary cloud; no multi-cloud pressure before Year 2.
- Gemini File Search remains usable at our scale and cost targets.
- Gemini 3 Flash and GPT-5-series families remain available and improve.
- We can acquire SOC 2 Type I within 9 months of starting M1.
- Stripe handles all targeted currencies.
- Customers accept cloud data processing (no contractual air-gap requirement in SMB segment).

### 17.3 Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | LLM cost exceeds projections | Med | High | Aggressive caching (semantic cache, response cache), model routing (cheap-first), quota enforcement, per-tenant cost dashboards. |
| R-02 | Cross-tenant data leak via prompt or search | Low | Catastrophic | RLS everywhere, per-tenant stores, pre-commit regression tests for tenant_id, red-team prompts in CI. |
| R-03 | Connector API rate limits throttle sync | Med | Med | Exponential backoff, per-source concurrency caps, delta sync preferred, priority queue per tenant. |
| R-04 | SSO/SCIM rollout slips enterprise deals | Med | High | Outsource to WorkOS/Auth0; tight M2 delivery. |
| R-05 | Agent runaway loops blow token budgets | Med | Med | Hard caps (max tool calls, max tokens, timeouts); cost-budget per run; alerting. |
| R-06 | Model provider outage | Low | High | Fallback model per agent; circuit breaker; queue requests. |
| R-07 | PII exposure in logs/telemetry | Med | High | PII detector in log pipeline; structured logging with allowlist fields only; training. |
| R-08 | Churn from Free-plan freeloaders without conversion | Med | Med | Tight free-plan limits; activation-based nudges; CS outreach at usage thresholds. |
| R-09 | Agent Builder too complex for non-technical users | Med | Med | Templates-first UX; wizards; strong test/eval tooling; starter gallery. |
| R-10 | Regulatory change (EU AI Act high-risk classification) | Med | Med | Transparency features built-in (source cites, model catalog, opt-out); monitor classifications; legal counsel. |

---

## 18. Appendix

### 18.1 Glossary

- **Tenant / Organization**: paying customer account; the primary isolation boundary.
- **Membership**: a user's role in a tenant.
- **Folder**: organizational container for documents within a tenant.
- **Document**: uploaded or connector-sourced file.
- **Parsed document**: Markdown + metadata produced by the parser.
- **Extracted record**: structured JSON produced by an extractor using a schema.
- **Schema**: JSON Schema defining fields for extraction (base fields + custom fields).
- **Template**: industry bundle (doc types + prompts + widgets + agents + samples).
- **File Search Store**: Gemini-hosted vector store used for RAG.
- **Connector**: integration module for a third-party SaaS (Drive, Slack, …).
- **Hybrid Query Router**: component that classifies a NL question and dispatches to SQL / RAG / connector / tool paths.
- **Agent**: user-defined LLM program with prompt, tools, knowledge, and guardrails.
- **Tool**: callable capability an agent can invoke (search, sql_query, send_email, custom MCP tool, …).
- **MCP**: Model Context Protocol, open standard for exposing tools to agents.
- **Knowledge Graph**: entity-relationship graph derived from a tenant's data.
- **RLS**: Row-Level Security (PostgreSQL policy engine).
- **PII**: Personally Identifiable Information.
- **CMEK**: Customer-Managed Encryption Key (via Cloud KMS).
- **SSO / SCIM**: Single Sign-On / System for Cross-domain Identity Management.
- **DSAR**: Data Subject Access Request (GDPR).

### 18.2 Mapping from v1 to v2 (at a glance)

| v1 Component | v2 Home | Notes |
|---|---|---|
| `doc_intelligence_ai_v3.0` | AI Backend (`biz2bricks-ai-v2`) | Strip auth/billing; keep agents, RAG, bulk, extraction, reports. |
| `doc_intelligence_backend_api_v2.0` | Merged / replaced | Partial absorb into Admin Backend (identity/billing) and AI Backend (proxy/BFF functions). |
| `agent_builder_v1` | Agent Builder feature in FE + AI | UI goes to `biz2bricks-fe-v2`; runtime in AI backend; metadata in Admin backend. |
| `document_intelligence_fe_v2` | Frontend (`biz2bricks-fe-v2`) | Upgrade to Next.js 15 + shadcn/ui + Tailwind; add search + agent builder + admin surfaces. |
| `biz2bricks_core` (ORM) | Keep and evolve | Schema split into `identity`, `billing`, `content`, `processing`, `search`, `agents`, `memory`, `audit`, `admin` namespaces. |
| v1 `src/agents/document/` | AI Backend `src/agents/document/` | Minor refactor; new `rag_search` → Unified Search. |
| v1 `src/agents/sheets/` | AI Backend `src/agents/sheets/` | Kept; consider merging with extraction long-term. |
| v1 `src/agents/extractor/` | AI Backend `src/agents/extractor/` | Keep; add human-in-the-loop review. |
| v1 `src/agents/report/` | AI Backend `src/agents/report/` | Keep; expand report templates. |
| v1 `src/bulk/` | AI Backend `src/bulk/` | Keep; extend to bulk agent runs. |
| v1 `src/rag/` | AI Backend `src/search/` (renamed) | Generalize to "unified search"; connectors plug in here. |
| v1 `src/core/usage/` | Admin Backend `src/billing/` | Quota enforcement moves to Admin; AI backend calls Admin. |
| v1 `src/db/repositories/audit/` | Admin Backend `src/audit/` | Centralize; AI emits events; Admin persists. |
| v1 `cloud_functions/bulk_trigger/` | Shared | Retained; publishes events to Pub/Sub. |

### 18.3 Example End-to-End Flows

**Flow A — Ask a hybrid question with connectors connected:**

```
User types:   "How much did we spend on Acme last quarter, and does
               their renewal clause have a price cap?"

Frontend → POST /api/v2/chat (SSE)  ─────┐
                                          │
                                          ▼
AI Backend /chat handler
  1. Resolve session, load memory.
  2. Admin.check_quota(org_id, 'chat')    (cached 60s; 429 if exceeded)
  3. HybridQueryRouter.classify(question)
       → {route: "hybrid",
          sql_question: "total spend on Acme in Q3",
          rag_question: "Acme contract renewal price cap"}
  4. Parallel execute:
       a. SQL path:
          - schema_registry.describe(org_id)
          - text_to_sql.generate(question, schema_desc)
          - sqlglot.validate(sql) → only SELECT, tenant_id filter, LIMIT
          - execute with statement_timeout=5s
          - rows = [...]
       b. RAG path:
          - select_stores(org_id, question) → ['{org}:uploads', '{org}:connector:google_drive']
          - file_search.query(stores, question)
          - connector_search.query(org, 'google_drive', question)
          - merge by RRF, return top-10 with citations
  5. Synthesizer.combine(sql_result, rag_result) via Gemini 3 Flash
  6. Stream tokens + citations via SSE.
  7. Emit event: chat.message.sent {org, user, tokens, cost}
  8. Persist message + trace to memory schema.
```

**Flow B — Bulk extraction via API:**

```
Client POST /api/v2/bulk/upload  (multipart)
  → documents staged to gs://.../orgs/{org}/uploads/{job}/
  → create bulk_job (LangGraph state machine)
  → for each doc (concurrency=3):
       parse → (auto-detect doc_type) → extract with schema → persist record
  → emit content.document.uploaded / parsed / extracted events
  → client polls /bulk/jobs/{id} or subscribes to webhook
  → on complete, export CSV/XLSX ready for download
```

**Flow C — Agent triggered by new-file-in-connector:**

```
Google Drive push notification → webhook → AI /internal/events/connector-sync
  → connector_sync enqueues delta for that folder
  → for each new doc:
       detect agents registered for trigger "connector.doc.added:google_drive:{folder}"
       for each agent:
         enqueue agent_run (Cloud Tasks)
         Agent executor:
           - load agent version, build LangGraph
           - tools = resolved from registry (scoped to agent)
           - input = {document_id, metadata}
           - run with guardrails (max_tokens, timeout, max_tool_calls)
           - persist agent_runs row + trace
           - emit agent.run.completed event
```

### 18.4 Reference Architecture for the Agent Runtime

```
AgentRequest → AgentOrchestrator
                 ├── resolve version & guardrails
                 ├── build LangGraph(tools, prompt, model)
                 ├── trace_ctx = opentelemetry.start_span
                 ├── loop:
                 │     LLM.call(state) → may propose tool call
                 │     validate tool call (allowed list + scopes)
                 │     tool.execute(args) with per-tool timeout
                 │     append result to state
                 │     check budget (tokens, cost, wall time, max_tool_calls)
                 ├── final: assemble output per output_schema
                 ├── persist agent_run with full trace
                 └── emit agent.run.completed event
```

### 18.5 Reference Stack Table

| Concern | Choice | Notes |
|---|---|---|
| Cloud | Google Cloud Platform | v1 is on GCP; continue. |
| Compute | Cloud Run (services + jobs) | Autoscale; min-instances for warmth. |
| Primary DB | Cloud SQL PostgreSQL 16 (HA) | RLS on every tenant-scoped table. |
| Vector / RAG | Gemini File Search (primary); pgvector as fallback for metadata embeddings | Re-evaluate Pinecone/Turbopuffer if scale demands. |
| Object storage | Google Cloud Storage | Per-org prefixes; KMS-CMEK for Enterprise. |
| Queue | Pub/Sub | Event-driven; Cloud Tasks for scheduled. |
| Cache / session | Memorystore Redis | Namespaced per org. |
| Scheduler | Cloud Scheduler | Agent schedules, connector sync. |
| Secrets | Secret Manager | Rotation automation. |
| CDN/WAF | Cloud CDN + Cloud Armor | DDoS and WAF. |
| Identity | WorkOS (primary pick) or Auth0 | SSO + SCIM. |
| Billing | Stripe | Metered + subscription. |
| Observability | OpenTelemetry → Cloud Trace / Cloud Monitoring / Cloud Logging; BigQuery archive | LLM-specific dimensions. |
| Analytics | BigQuery | Usage rollups, search analytics. |
| CI/CD | GitHub Actions + Artifact Registry | Blue/green on Cloud Run. |
| IaC | Terraform | Per-env workspaces + regional modules. |
| Feature flags | LaunchDarkly or Unleash | Per-org targeting. |
| PII detection | Microsoft Presidio or Google DLP | Configurable strategy. |

---

**End of BRD v1.0 Draft.**

*Next actions:* circulate for engineering review; lock M1 scope; stand up the three repos (`biz2bricks-ai-v2`, `biz2bricks-admin-v2`, `biz2bricks-fe-v2`) with scaffolded directories matching §8.4 and §9.1; begin Claude Code / Codex-led implementation from the sequenced backlog in §15.3.
