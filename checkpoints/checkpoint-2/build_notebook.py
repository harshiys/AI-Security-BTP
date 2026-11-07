import nbformat as nbf
src = lambda p: open(p, encoding="utf-8").read().rstrip("\n") + "\n"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
files = ["cp2_contract.py", "metrics_extra.py", "simulate_logs.py", "mla_pipeline.py", "trust_layer.py", "graph_features.py",
         "ml_b_eval.py", "cybersec_adapters.py", "scorer.py", "tamas_inspect.py", "tamas_run_subset.py"]
cells = [
md("""# SentinelMesh CP2: ML-A + ML-B with real TAMAS data support

Four stages, kept apart. This notebook runs **1 (synthetic self-test, RoundRobin only to stay quick), 2 (TAMAS data inspection)** and the **dry run of 3**.
Stage 3 (real LLM execution) and stage 4 (real TAMAS-derived ML) need an OpenAI key and internet: see the last section.

**Every number from stage 1 is a pipeline self-test on fake logs with invented labels. They are NOT TAMAS results.**
Read the README section *TAMAS Ground-Truth Labels and Limitations* before showing anything to anyone. About 10 minutes."""),
code("%pip install -q numpy pandas scikit-learn tabulate networkx jsonschema joblib"),
code("import os, subprocess\nif not os.path.isdir('TAMAS'):\n    print(subprocess.run(['git','clone','--depth','1','https://github.com/microsoft/TAMAS.git'],capture_output=True,text=True).returncode)\nos.makedirs('schema', exist_ok=True)\nos.makedirs('tests/fixtures', exist_ok=True)\nprint(sorted(os.listdir('TAMAS')))"),
md("### Write the code files and the team's shared schema"),
]
for f in files:
    cells.append(code(f"%%writefile {f}\n" + src(f)))
cells.append(code("open('schema/shared_schema_format.json','w',encoding='utf-8').write(" + repr(src("schema/shared_schema_format.json")) + ")\n"
                  "open('schema/sidecar.schema.json','w',encoding='utf-8').write(" + repr(src("schema/sidecar.schema.json")) + ")\nprint('schemas written')"))
cells += [
md("## Stage 2: TAMAS data inspection (read-only)"),
code("%run tamas_inspect.py --tamas_dir TAMAS --out reports"),
md("## Stage 1: synthetic pipeline self-test (fake logs, NOT TAMAS results)"),
code("%run simulate_logs.py --tamas_dir TAMAS --out synthetic_selftest/sim --configs RoundRobin"),
code("%run mla_pipeline.py --tamas_dir TAMAS --logs_dir synthetic_selftest/sim/logs --harmless_dir synthetic_selftest/sim/Harmless --out synthetic_selftest/out"),
code("import pandas as pd\nfrom IPython.display import display, Markdown\npd.set_option('display.width',220); pd.set_option('display.max_columns',40)\ndisplay(Markdown(open('synthetic_selftest/out/evaluation_table.md',encoding='utf-8').read()))"),
code("%run trust_layer.py --out synthetic_selftest/out"),
code("%run ml_b_eval.py --out synthetic_selftest/out"),
md("### The scorer: run prefix in, one record in the shared schema out"),
code("import json, scorer, cp2_contract\nbundle = scorer.load_bundle('synthetic_selftest/out/mlb_bundle.joblib')\nev = pd.read_csv('synthetic_selftest/out/events.csv'); lab = pd.read_csv('synthetic_selftest/out/labels.csv')\nrid = ev.run_id.iloc[0]; run = ev[ev.run_id == rid]\nrecord, sidecar = scorer.score_run_prefix(run, rid, bundle, roster=lab[lab.run_id==rid].agent.tolist(), task_id=run.task_id.iloc[0])\nprint(json.dumps(record, indent=1))\nbad = [l for l in open('synthetic_selftest/out/checkpoint2_verdicts.jsonl') if not cp2_contract.validate_record(json.loads(l))]\nprint('lines in checkpoint2_verdicts.jsonl that fail the shared schema:', len(bad))"),
md("## Tests"),
code("print('The tests live in tests/. Run from the package folder:  python -m unittest discover -s tests -t . -v')"),
md("""## Stage 3 and 4: real TAMAS data (needs an OpenAI key and internet)
First the dry run (prints the exact commands and runs nothing):"""),
code("%run tamas_run_subset.py --tamas_dir TAMAS --attack DPI --scenario news --config RoundRobin --model gpt-4o --n 3 --out tamas_real --dry_run"),
md("""Then, on your own machine:
```
export OPENAI_API_KEY=sk-...
python tamas_run_subset.py --tamas_dir TAMAS --attack DPI --scenario news --config RoundRobin --model gpt-4o --n 10 --out tamas_real
python mla_pipeline.py --tamas_dir TAMAS --logs_dir tamas_real/logs --out out_tamas_real
python trust_layer.py --out out_tamas_real
python ml_b_eval.py   --out out_tamas_real
```
In this session stage 3 was blocked (no credentials; the sandbox cannot reach the API). See `reports/real_execution_status.md`."""),
]
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, "SentinelMesh_CP2_MLA_MLB.ipynb")
print("built", len(cells))
