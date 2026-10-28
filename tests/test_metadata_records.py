"""Fresh-process metadata replay without importing models or loading assets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("study_io", "campaign_runtime")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/ogo" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record():
    return {"identity": "fixture", "model": {"preprocess": {
        "mean": (.485, .456, .406), "std": (.229, .224, .225)}},
        "targets": ["attention.qkv", "attention.proj"]}


class MetadataRecords(unittest.TestCase):
    def test_fresh_process_resume_accepts_serialized_tuples(self):
        script = (
            "import importlib.util,json,sys; "
            "s=importlib.util.spec_from_file_location('records',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "v=json.loads(sys.argv[3]); "
            "v['model']['preprocess']['mean']=tuple(v['model']['preprocess']['mean']); "
            "v['model']['preprocess']['std']=tuple(v['model']['preprocess']['std']); "
            "m.write(sys.argv[2],v,immutable=True)"
        )
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "identity.json"
                load(name).write(path, record(), immutable=True)
                original = path.read_bytes()
                subprocess.run([sys.executable, "-B", "-c", script,
                                str(ROOT / "src/ogo" / (name + ".py")),
                                str(path), json.dumps(record())], check=True)
                self.assertEqual(path.read_bytes(), original)

    def test_changed_scientific_identity_is_rejected(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as tmp:
                module = load(name)
                path = Path(tmp) / "identity.json"
                module.write(path, record(), immutable=True)
                original = path.read_bytes()
                changed = record()
                changed["model"]["preprocess"]["mean"] = (.5, .456, .406)
                with self.assertRaises(ValueError):
                    module.write(path, changed, immutable=True)
                self.assertEqual(path.read_bytes(), original)

    def test_invalid_metadata_does_not_damage_committed_record(self):
        for name in MODULES:
            for bad in (float("nan"), float("inf"), object()):
                with self.subTest(module=name, bad=type(bad).__name__), tempfile.TemporaryDirectory() as tmp:
                    module = load(name)
                    path = Path(tmp) / "identity.json"
                    module.write(path, record(), immutable=True)
                    original = path.read_bytes()
                    with self.assertRaises((ValueError, TypeError)):
                        module.write(path, {"value": bad})
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
