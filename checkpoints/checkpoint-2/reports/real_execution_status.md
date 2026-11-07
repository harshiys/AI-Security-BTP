# Stage 3: actual TAMAS execution: BLOCKED in this session

Nothing was executed against an LLM, no real logs exist, and none were invented.

Measured in the session where this package was prepared:
- `OPENAI_API_KEY` and `AZURE_OPENAI_API_KEY` are not set; `ollama` is not installed.
- A request to `api.openai.com:443` was refused by the sandbox's egress proxy (curl status 000, proxy reported `connect_rejected`). It was not retried or worked around.
- `autogen-agentchat 0.7.5` is installed, so the framework itself is not the blocker.
- `tamas_run_subset.py ... --dry_run` works and prints the exact commands. Without a key it stops with exit code 2 and writes the status file (`reports/real_execution_status_attempt.json`).

To continue on a machine with an OpenAI key and internet:
```bash
git clone --depth 1 https://github.com/microsoft/TAMAS.git TAMAS
pip install autogen-agentchat "autogen-ext[openai]" numpy pandas scikit-learn tabulate networkx jsonschema joblib
export OPENAI_API_KEY=sk-...
python tamas_run_subset.py --tamas_dir TAMAS --attack DPI --scenario news --config RoundRobin --model gpt-4o --n 10 --out tamas_real
python mla_pipeline.py --tamas_dir TAMAS --logs_dir tamas_real/logs --out out_tamas_real
python trust_layer.py --out out_tamas_real
python ml_b_eval.py   --out out_tamas_real
```
Also run `--config Swarm` and `--config Magentic_one` for the three TAMAS team configurations. Ten DPI news tasks per configuration is
the most the unmodified repo supports (see README, "Problems found in the TAMAS repo"). Cost: roughly 10 short multi-agent chats per configuration.
