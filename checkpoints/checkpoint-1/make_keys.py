"""Creates scanner_private.pem / scanner_public.pem (Ed25519). Run once. Never share or commit the private key."""
import os
from cryptography.hazmat.primitives import serialization as s
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

if os.path.exists("scanner_private.pem"):
    raise SystemExit("scanner_private.pem already exists; not overwriting")
k = Ed25519PrivateKey.generate()
open("scanner_private.pem", "wb").write(
    k.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
open("scanner_public.pem", "wb").write(
    k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
print("keys written")
