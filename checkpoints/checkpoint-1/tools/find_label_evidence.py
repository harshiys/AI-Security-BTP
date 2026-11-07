#!/usr/bin/env python3
"""Show the lines of the OFFICIAL PEFTGuard code that could tell us what label0 / label1 mean.

It fetches train.py, contrast_train.py, run.sh and README.md from the authors' GitHub repo and prints every line that
mentions labels, benign/backdoor/poison/clean, safetensors, or folder listing - with a few lines of context.
It DECIDES NOTHING. You read the lines. If a line clearly shows how label0/label1 become 0/1 and which of those means
'backdoored', copy the file name + line into the "evidence" field of work/label_rules.json.

    python tools/find_label_evidence.py            # needs internet (GitHub)
    python tools/find_label_evidence.py --file some_local_copy.py     # offline: scan a file you saved
"""
import argparse, re, sys, urllib.request

BASE = "https://raw.githubusercontent.com/Vincent-HKUSTGZ/PEFTGuard/main/"
FILES = ["train.py", "contrast_train.py", "run.sh", "README.md"]
PATTERN = re.compile(r"label\s*[_ ]?\s*[01]\b|label[01]|\blabels?\b|benign|backdoor|poison|clean|safetensors|"
                     r"os\.listdir|glob|os\.walk|adapter_model", re.I)


def scan(name: str, text: str, ctx: int = 2) -> int:
    lines = text.splitlines()
    hit = [i for i, l in enumerate(lines) if PATTERN.search(l)]
    print(f"\n===== {name}: {len(lines)} lines, {len(hit)} matching lines =====")
    shown = set()
    for i in hit:
        for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
            if j not in shown:
                shown.add(j)
                print(f"{'>>' if j in hit else '  '} {j+1:4d}: {lines[j][:200]}")
        print("  ----")
    return len(hit)


ap = argparse.ArgumentParser()
ap.add_argument("--file", help="scan a local file instead of fetching from GitHub")
args = ap.parse_args()
total = 0
if args.file:
    total += scan(args.file, open(args.file, encoding="utf-8", errors="replace").read())
else:
    for f in FILES:
        try:
            text = urllib.request.urlopen(BASE + f, timeout=20).read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"\n===== {f}: FETCH FAILED ({type(e).__name__}: {e}) =====")
            continue
        total += scan(f, text)
print("\n" + "=" * 70)
print(f"{total} matching lines in total. This script proves nothing by itself.")
print("If the code does NOT make the label0/label1 meaning clear, ask the authors (GitHub Issues on\n"
      "Vincent-HKUSTGZ/PEFTGuard) and quote their answer in the 'evidence' field. Do not guess.")
