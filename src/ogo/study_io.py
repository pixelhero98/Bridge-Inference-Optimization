"""Portable immutable identities and auditable study records."""
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value, *, immutable=False):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        # JSON maps tuples to lists; compare the representation stored on disk.
        if read(path) != json.loads(payload):
            raise ValueError('Immutable identity changed: '+str(path))
        return
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(payload, encoding='utf-8')
    os.replace(temporary, path)


def sources():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
