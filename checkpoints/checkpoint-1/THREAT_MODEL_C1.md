# Checkpoint 1 Threat Model: Model/Adapter Supply Chain

## Assets
- The LoRA/PEFT adapter files that will be loaded into a deployment
- The scanner's verdicts (pass / flag / block) and scores
- The scanner's private signing key
- The scanner's trained classifier file
- The public key / key fingerprint used by the deployment gate
- The training data and labels (PADBench or equivalent)

## Attacker
- Can publish a backdoored adapter to a public model hub
- Can modify files in transit or at rest between scanning and deployment
- Can read and write the artifact store and the signed-record store
- Cannot read the scanner's private key (assumed; see residual risks)
- May know the scanner's design and features (adaptive attacker)

## Attack paths and defenses
| # | Path | Defense | Status |
|---|------|---------|--------|
| 1 | Upload backdoored adapter that looks benign | Spectral-feature classifier | Partial (detection quality dependent; no real PADBench result yet) |
| 2 | Swap file after it passed the scan | Gate re-hashes file at load time | Covered |
| 3 | Edit verdict or score in the record | Ed25519 signature over full record | Covered |
| 4 | Swap file and rewrite hash in the record | Signature covers the hash | Covered |
| 5 | Forge a record with attacker's own key | Gate verifies with the scanner's public key | Covered |
| 6 | Replace the public key file with attacker's key | Key pinning (--pin key_id) | Covered only if the pin is stored where attacker cannot write |
| 7 | Reuse an old pass later | 24h freshness window, future-timestamp rejection | Covered |
| 8 | Reuse a pass record for another file | Hash bound to one file; item_id check | Covered |
| 9 | Replace the scanner with a weaker one | Classifier file sealed; loaded only through verified_load.py | Implemented, tested on synthetic self-test only |
| 10 | Craft adapter that evades classifier on purpose | Adaptive-attack evaluation | Pending (needs final classifier) |
| 11 | Scan one file, sign another | Scan-time sha256 recorded in s2; sign_c1.py refuses mismatches | Implemented, tested on synthetic self-test only |

## Residual risks (stated limitations)
1. Private key theft lets an attacker sign anything. Production use needs an HSM or a signing service.
2. Key pinning only helps if the pinned value is stored outside the attacker's write access.
3. A valid signature proves the scanner issued the verdict, not that the verdict was correct. Chain strength equals detection strength.
4. Binding a verdict to the scanned file holds only if the scan-time hash is recorded; without it the signature covers the file as found.
5. No revocation: a signed pass stays valid until it expires, even if the adapter is later found malicious.
6. Clock trust: freshness depends on the verifier's system clock.
7. The private key is stored unencrypted on disk in this prototype.

## Out of scope
- Attacks on the base model itself
- Runtime behavior of the loaded adapter (this is Checkpoint 2)
- Compromise of the machine running the gate
