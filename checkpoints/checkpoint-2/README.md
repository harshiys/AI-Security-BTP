# SentinelMesh Checkpoint 2: Runtime Multi-Agent Trust & Behavioral Fabric

**Milestone**: Checkpoint 2 — Runtime Agent Identity & Behavioral Trust Monitoring  
**Dataset**: 100% Real TAMAS (Microsoft AutoGen Direct Prompt Injection Benchmark)  
**Schema Compliance**: Strict Draft-7 5-Field Contract (`checkpoint_id: "CKPT2_RUNTIME"`)  

---

## 1. System Overview

Checkpoint 2 provides the runtime behavioral and identity monitoring layer for multi-agent LLM systems. Once an agent’s model weights pass Checkpoint 1, Checkpoint 2 monitors the live interactions between agents as they collaborate, invoke tools, and exchange messages.

### The Two-Layer Defense:
1. **Cyber Telemetry Layer (`cyber_src/`)**:
   - **Behavioral Fingerprinting**: Tracks capability and tool-calling drift away from the agent's baseline.
   - **Communication Graph**: Analyzes coordination topology (in/out degree, PageRank, betweenness, novel interaction edges).
   - **Message Provenance**: HMAC-SHA256 protocol implementation and historical TAMAS log audit.
   - **Dynamic Trust Engine**: Computes continuous trust $T = 1 - \text{cyber\_risk}$.
2. **Machine Learning Behavioral Layer (`out_tamas_real/`)**:
   - **ML-A**: Per-agent behavioral feature extraction across decision points $K \in \{1, 2, 4, 8\}$.
   - **ML-B**: Inter-agent interaction graph features + calibrated Logistic Regression / Random Forest models.
   - **Trust Scoring Layer**: Out-of-fold threshold calibration ensuring a 5% false-positive budget.

---

## 2. Dataset Provenance: Real TAMAS

* **Benchmark**: Microsoft TAMAS (Trustworthy Agentic Multi-Agent System Benchmark)
* **Architecture**: Multi-agent AutoGen pipelines (News gathering, Fact checking, Article writing, Distribution).
* **Attacks Evaluated**: Direct Prompt Injection (DPI) attacks where adversarial prompts attempt to force agents into invoking unauthorized external tools.
* **Scale**: 10 authentic full execution runs, 40 agent instances.
* **Provenance Guarantee**: `data/events.csv` and `data/labels.csv` contain authentic execution traces with zero synthetic data.

---

## 3. Strict 5-Field Schema Output

Every verdict record exported to `results/checkpoint2_verdicts.jsonl` adheres strictly to `schema/shared_schema_format.json`:

```json
{
  "checkpoint_id": "CKPT2_RUNTIME",
  "item_id": "50b83d18",
  "verdict": "pass",
  "score": 0.4963,
  "timestamp": "2026-10-27T13:29:14.204712+00:00"
}
```
* Exactly 5 fields (`additionalProperties: false`).
* Internal mapping between `item_id` and `(run_id, agent)` is preserved in `out_tamas_real/cp2_item_index_internal.csv` and `results/checkpoint2_fused_summary.csv`.

---

## 4. Quick Start Commands

### 1. Execute the Master Integrated Pipeline
```powershell
python run_checkpoint2.py
```
> Runs Cyber telemetry $\to$ ML-B scoring $\to$ Cyber-ML fusion $\to$ 100% Draft-7 schema validation.

### 2. Live Runtime Gatekeeper CLI (`verify_cp2.py`)
```powershell
# Verify a specific clean agent
python verify_cp2.py --run-id r71a932976c --agent article_writing_agent

# Verify a flagged anomaly agent
python verify_cp2.py --run-id r77c0e4eb97 --agent news_gathering_agent

# Test Controlled Cyber Attack Scenarios (Red-Team)
python verify_cp2.py --scenario impersonation
python verify_cp2.py --scenario rogue_delegation
python verify_cp2.py --scenario sleeper_agent

# Display the full runtime decision matrix for all 40 agents
python verify_cp2.py --all
```

### 3. Run Automated Integration Test Suite
```powershell
python tests/run_all_cp2_tests.py
```
> Runs 9 automated tests verifying dataset provenance, schema compliance, dynamic trust math, red-team attack mitigation, and gatekeeper execution.
