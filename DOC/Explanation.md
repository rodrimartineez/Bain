# Claim Triage Assistant — technical explanation

## 1. Problem

An insurtech company wants to help adjusters process incoming property-damage claims faster and more
consistently. A claim arrives with a free-text description, damage photos and supporting documents. The system
must analyse the submission, research similar past claims and produce a **structured triage report**. The
assignment requires a **multi-agent system with a ReAct loop**, a **vector knowledge base of historical claims**
and a **lightweight web UI**.

**Missing materials.** The Bain assessment folder was extracted without files: every sub-folder
(`api-contract`, `auto-grader/tests`, `data/knowledge-base`, `ground-truth`, `test-images`, `test-documents`,
`tools`) is empty. I therefore treated `docs/correo.md` as the specification and built everything else myself:
an internal API contract, a synthetic knowledge base, synthetic demo inputs and my own tests and evaluation set.
I make no claim of compatibility with Bain's contract or auto-grader.

## 2. Architecture at a glance

- **Python + FastAPI** backend, **vanilla JS** single-page UI, **Ollama** for all models (local, free, no keys).
- **Five agents** with clear responsibilities; one of them, the **Coordinator**, runs the ReAct loop.
- **Vector knowledge base**: 40 synthetic historical claims embedded with `nomic-embed-text`, exact cosine search.
- **Pydantic** schemas for every intermediate result and for the final report.

```
UI ─► POST /api/claims ─► intake validation ─► background job ─► Coordinator (ReAct)
                                                                  ├─ IntakeAnalyst   (description / photo / document tools)
                                                                  ├─ KnowledgeResearcher (vector search)
                                                                  ├─ EvidenceAuditor (consistency checks)
                                                                  └─ TriageAssessor  (priority, confidence, narrative)
UI ◄─ GET /api/claims/{job} (live trace + validated TriageReport)
```

### Why these technologies

| Decision | Reason |
|---|---|
| Python | Best ecosystem for LLM, PDF and image work; fastest to build and test. |
| FastAPI | Typed multipart handling, automatic OpenAPI (`/docs`), trivial background jobs. |
| Local Ollama models | Available on the machine, zero cost, no secrets in the repo, reproducible. I checked availability and speed before relying on them. |
| `qwen2.5:3b` for text | It is the only text model that fits entirely in the 4 GB GPU next to the other apps: ~8 s per call. `gemma3:4b` does not fit and was 10–50× slower for text. |
| `gemma3:4b` for photos | The only local model that actually reads images. Run on CPU (`VISION_NUM_GPU=0`) on this laptop: ~50 s per photo, versus >10 min with a partial GPU split. |
| `nomic-embed-text` | Small, strong retrieval embedding model; uses `search_document:` / `search_query:` prefixes. |
| NumPy exact search instead of FAISS/Chroma | For tens to thousands of records an exact matrix–vector product is instant, deterministic and dependency-free (FAISS/Chroma wheels for Python 3.14 were also a risk). The `VectorIndex` class is the only place that would change for FAISS/pgvector. |
| Vanilla JS UI | No build step, nothing to break during a demo; it only renders what the backend returns. |

## 3. The multi-agent system

Each agent owns tools and a single responsibility (`claim_triage/agents/`):

| Agent | Tools | What it does | LLM? |
|---|---|---|---|
| **IntakeAnalystAgent** | `analyze_description`, `analyze_image`, `extract_document` | Turns raw inputs into structured facts. Description → declared facts (damage type, cause, hazards, habitability, ongoing damage, loss). Photo → visible damage, severity, hazards, whether the image looks synthetic. Document → text via pypdf, amounts/dates via regex, type + key facts via LLM. | yes |
| **KnowledgeResearcherAgent** | `search_similar_claims` | Builds a query from *all* evidence so far (declared + visual), searches the vector index, explains each hit and aggregates outcomes of precedents. | embeddings only |
| **EvidenceAuditorAgent** | `cross_check_evidence` | Deterministic checks: declared vs visible damage, photos actually show damage, photo authenticity, claimed amount vs documents, date sanity, documents dated before the incident, red flags in close precedents, unprocessed inputs. | no (on purpose) |
| **TriageAssessorAgent** | `assess_claim` | Deterministic priority rubric and confidence indicator; LLM writes the narrative, which is validated and screened. | narrative only |
| **CoordinatorAgent** | `finish` | Runs the ReAct loop and decides which tool to use next. | yes |

Consistency checks and priority scoring are deterministic on purpose: they must be reproducible and explainable,
which an LLM judgement is not. The LLM is used where language or vision understanding is needed.

## 4. The ReAct loop (`claim_triage/agents/coordinator.py`)

```
for step in 1..MAX_REACT_STEPS (12):
    options  = valid (action, target) pairs from tool preconditions      # observe state
    decision = policy.decide(state, options)                             # REASON: LLM picks one + short reason
    if decision is finish: build + validate report, stop
    observation = tool.run(state, target)                                # ACT
    record TraceStep(agent, action, target, reason, decided_by, status, observation, duration)   # OBSERVE
```

- **Real tool selection.** The LLM receives a compact state summary (pending images/documents, damage types so
  far, best similarity, whether search/check/assessment are stale, warnings, errors), the last observation and
  the menu of allowed actions. It answers JSON `{reason, action, target}`. In practice it chooses different
  orders (e.g. image before description, or document before search). Unit tests prove the planner's choice
  is executed.
- **Preconditions** come from each tool: search needs an analysed description; assessment needs at least one
  search and an up-to-date cross-check; finish needs an up-to-date assessment.
- **State staleness.** Every new piece of evidence increments `evidence_version`. A search, cross-check or
  assessment made on an older version becomes available again; this is how the agent decides to *re-search*
  after a photo reveals new information.
- **Guardrails.** The schema `enum` limits the model to the allowed actions; the choice is validated again.
  If the action is valid but the target is wrong (seen in practice: `analyze_description` with an image
  name), the target is repaired and marked `[guardrail …]`. Otherwise, or if the LLM call fails, the
  deterministic `RulePolicy` decides (`decided_by = fallback_rules`). When only one action is valid, no LLM call
  is made (`decided_by = rules`). Max 2 searches, max 2 assessments, failed searches are retried once.
- **Controlled termination.** At the step limit the run stops, a report is still built from the available
  evidence with a template narrative, `termination = max_steps_reached`, and human review is required.
- `COORDINATOR_MODE=rules` runs the same loop with the deterministic planner only (useful as a baseline and
  when no LLM is available for decisions).

The trace stores *operational* reasons (why this action now), not private chain-of-thought.

## 5. Vector knowledge base (`claim_triage/knowledge_base.py`)

- **Data**: `data/knowledge_base/historical_claims.json` — 40 **synthetic** claims (every record has
  `"synthetic": true`) covering water leak, flood, storm/wind, hail, roof, fire, smoke, structural, mould and
  vandalism. Each has an ID, description, cause, property type, location, evidence list, severity, assigned
  priority, estimated/settled amounts, outcome (approved / partially approved / denied / referred to
  investigation), resolution notes, days to close and flags (e.g. `gradual_damage`, `fraud_indicators`,
  `electrical_hazard`). Several are deliberately "hard" (denied for wear and tear, inconsistent evidence,
  SIU referral) so that precedents carry useful warnings.
- **Representation**: each claim is rendered to a canonical text (damage type, cause, property, description,
  evidence, severity) and embedded with the `search_document:` prefix.
- **Index**: L2-normalised float32 matrix saved to `data/index/vectors.npy` plus `manifest.json` (model,
  dimensions, dataset hash, IDs). It is rebuilt automatically if the data or the embedding model change.
- **Search**: the query embedding (`search_query:` prefix) is multiplied by the matrix → cosine similarity, top-5,
  minimum similarity **0.63**. I chose the threshold by measuring: relevant queries score 0.64–0.92, off-topic
  queries scored ≤ 0.61.
- **Evidence shown**: claim ID, similarity score, type, severity, outcome, settled amount, days to close, flags and
  a "why similar" explanation (same damage type, same property type, shared concepts). The explanation is a
  separate layer; ranking is purely by embedding similarity. Aggregated *historical insights* (outcome
  distribution, frequent flags, settlement range, median time to close) feed the auditor and the report.

## 6. Input analysis

- **Intake validation** (`claim_triage/ingestion.py`): description ≥ 20 characters; max 5 images (jpg/png/webp)
  and 5 documents (pdf/txt/md), ≤ 10 MB each; unsupported or empty files → HTTP 422 with every error listed in the
  UI. Corrupt or tiny images are accepted with a warning so that the rest of the claim can still be triaged;
  the tool then records the failure as a limitation.
- **Description**: LLM with a JSON schema and explicit definitions (e.g. "uninhabitable only if the whole home…").
  **Grounding**: an extracted amount or date that does not appear in the text is dropped and listed in the
  report's uncertainties.
- **Photos**: the vision model returns visible damage, damage types, observations, severity, hazards and
  `image_kind`. It is told to describe only what is visible. It correctly flags our drawn demo images as
  `illustration_or_synthetic`, which the auditor reports.
- **Documents**: pypdf text extraction; regex for EUR amounts (`4,350.00`, `4.350,00`, `EUR 4350`, `780 €`) and
  dates; LLM for document type and key facts. PDFs without a text layer are reported as "no extractable text
  (no OCR)".

## 7. Triage report (`claim_triage/schemas.py::TriageReport`)

The report is a Pydantic model (schema version 1.0) shown in the UI and downloadable as JSON. Key sections:
executive summary, primary/secondary damage type, **facts with category and source** (`declared`, `observed`,
`document`, `historical`, `inferred`), photo and document findings, similar claims with similarity and
reasons, historical insights, consistency checks, preliminary assessment (labelled *inference*), priority,
recommended actions (labelled *recommendation*), information to request, uncertainties, confidence, human
review flag and reasons, narrative source, termination status, full tool trace and a disclaimer.

**Priority** (deterministic rubric, each factor with its source):
safety hazard +3 · uninhabitable +3 (partially +1) · ongoing damage +2 · visible severity critical +3 / severe +2 /
moderate +1 (if no photo: closest precedent's severity −1, labelled *historical (inferred)*) · loss ≥ 25k EUR +2 /
≥ 5k +1 · vulnerable occupants +1.
**P1 Critical ≥ 6** (emergency response within 24 h) · **P2 High 4–5** (contact within 48 h) ·
**P3 Medium 2–3** (standard queue) · **P4 Low 0–1** (fast-track/desk review).

**Confidence** = mean of five components in [0,1]: input coverage, evidence agreement (share of passed checks),
precedent support (from best similarity), fact completeness and presence of visual evidence → low / medium / high.
It is explicitly labelled as a heuristic, **not a calibrated probability**. Contradictory evidence caps the
level: *low* if fewer than half of the applicable checks pass, at most *medium* if any check fails.

**Human review** is recommended for P1 claims, failed checks, key warnings (declared vs visual, amount
mismatch, document dates, precedent red flags, photo authenticity, inputs that could not be processed), low
confidence or early termination.

**Output hygiene**: placeholder values such as `"unknown"` are removed from extracted lists (otherwise
`safety_hazards: ["unknown"]` would score +3), and repeated sentences are removed from the narrative.

**Narrative safety**: the LLM narrative must pass the schema (lengths, list sizes), must not cite claim IDs that
were not retrieved and must not contain coverage/payment decisions ("claim is approved", "we will pay", …).
Failure → one retry with the error fed back → deterministic template (`narrative_source = template_fallback`).
The system never decides coverage, liability or indemnity.

## 8. How to run

```bash
ollama pull qwen2.5:3b && ollama pull gemma3:4b && ollama pull nomic-embed-text
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python run.py                    # http://localhost:8000
```

CLI: `python scripts/run_case.py data/demo/01_ceiling_leak`. Settings: `claim_triage/config.py`, `.env.example`.
See `DOC/Demo_Instructions.md` for the demo.

## 9. Testing (what was actually run)

| Check | Command | Result |
|---|---|---|
| Unit + API tests (fake models) | `pytest` | **52 passed** |
| Integration tests (real Ollama) | `pytest -m ollama` | **6 passed** |
| Lint / format | `ruff check .`, `ruff format --check .` | clean |
| Retrieval evaluation (12 paraphrased queries, own data) | `python scripts/run_eval.py` | hit@3 = 1.00, MRR@5 = 1.00, damage-type precision@3 = 0.72, off-topic queries rejected 3/3 |
| End-to-end evaluation (4 demo cases, LLM coordinator) | `python scripts/run_eval.py --e2e` | **4/4 cases** pass all their checks (completed, damage type, priority range or review flag, expected precedent retrieved); 6–31 s per case in the final run; 0 fallback decisions (`output/eval_results_e2e.json`) |
| Manual UI test in the browser | — | empty description error, demo case load, live trace, full report rendered |

Unit tests cover: full flow; LLM-driven tool choice being executed; invalid action → fallback; planner outage;
wrong-target repair; no unnecessary repetition; re-search after new evidence; step-limit termination; vision tool
failure; corrupt image and text-less PDF; vector-search outage; rules mode; priority rubric (P1 and P4 cases);
hallucinated amount dropped; narrative with bad JSON / coverage decision / invented claim ID / LLM outage;
inconsistent evidence flags; schema rejection of invalid priority; intake validation; HTTP API errors and job
lifecycle; path-traversal protection on demo files.

The evaluation sets are small and self-made; they show the pipeline behaves as designed, they are **not** a
benchmark on real claims.

## 10. Limitations

- Small local models: extraction mistakes happen (observed: `ongoing_damage = true` although the water was shut
  off; the vision model sometimes adds "cracks" that are not there). They remain visible because every
  priority point has a source and review flags are raised.
- Latency depends heavily on hardware: ~30–70 s per multimodal claim on the development laptop (GPU shared with
  other applications), most of it in the vision call.
- Synthetic data only; precedent statistics are illustrative. Demo images are drawings, not photographs.
- No OCR, no authentication, in-memory job store, one worker, English-only prompts.
- Bain's API contract, ground truth and auto-grader were unavailable, so compatibility is unknown.

## 11. Most important decisions

1. **LLM decides the flow, rules guard it.** Real ReAct tool selection with preconditions, schema-constrained
   choices, deterministic fallback and a hard step limit: flexible but never stuck or looping.
2. **Deterministic priority and checks, LLM only for language and vision.** Same evidence → same priority,
   and every point is traceable.
3. **Facts are labelled by origin** (declared / observed / document / historical / inferred) and unsupported
   numbers are removed.
4. **Exact vector search with a persisted, self-invalidating index**: simplest correct option at this scale.
5. **Fully local stack** chosen after measuring the hardware, with the slow part (vision) isolated and configurable.
6. **Honesty by design**: synthetic data is labelled everywhere, confidence is not presented as a probability,
   and the report never decides coverage.

**Production next steps**: hosted/larger models behind `LLMClient`, OCR, a vector DB with metadata filters,
persistent queue and storage, authentication/PII controls, calibration of the priority rubric against
adjuster decisions, and an evaluation set of real labelled claims.
