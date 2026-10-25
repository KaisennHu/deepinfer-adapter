"""Exercise real shell launchers with a recording executable, without GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.recorder = self.directory / "python"
        self.recorder.write_text(
            f"#!{sys.executable}\nimport json,os,sys\n"
            "print(json.dumps({'argv':sys.argv[1:],'env':dict(os.environ)}))\n"
        )
        self.recorder.chmod(0o755)
        self.env = {"PATH": f"{self.directory}:{os.defpath}",
                    "DEEPINFER_REPO": str(self.directory / "engine"),
                    "MODEL_PATH": str(self.directory / "model"),
                    "HANDOFF_DIR": str(self.directory / "handoff"),
                    "PD_CUDA_IPC_POOL_ID": "test-pool"}

    def launch(self, script, **env):
        return subprocess.run(["bash", str(ROOT / "scripts" / script)],
                              env=self.env | env, text=True, capture_output=True)

    def worker(self, role, profile="none"):
        result = self.launch("run_pd_http.sh", PD_ROLE=role, PD_PROFILE=profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        argv = record["argv"][3:]  # -u -m module
        return dict(zip(argv[::2], argv[1::2])), record["env"]

    def test_defaults_separate_ports_and_devices(self):
        p, pe = self.worker("prefill")
        d, de = self.worker("decode")
        self.assertEqual((p["--port"], d["--port"]), ("18300", "18400"))
        self.assertEqual((p["--device-offset"], d["--device-offset"]), ("0", "4"))
        ports = {int(pe["MASTER_PORT"]), int(pe["MASTER_PORT"]) - 50,
                 int(de["MASTER_PORT"]), int(de["MASTER_PORT"]) - 50}
        self.assertEqual(len(ports), 4)

    def test_validated_profile_keeps_compatible_handoff_and_role_specific_state(self):
        p, pe = self.worker("prefill", "b300-1p1d")
        d, de = self.worker("decode", "b300-1p1d")
        for key in ["--nextn", "--max-model-len", "--dp-size", "--handoff-dir"]:
            self.assertEqual(p[key], d[key])
        self.assertEqual(p["--nextn"], "7")
        self.assertEqual(pe["PD_CUDA_IPC_POOL_ID"], de["PD_CUDA_IPC_POOL_ID"])
        self.assertEqual(pe["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"], "compact")
        self.assertEqual(de["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"], "off")
        self.assertEqual(pe["DEEPINFER_FUSE_GREEDY_REDUCTION"], "0")
        self.assertEqual(de["DEEPINFER_FUSE_GREEDY_REDUCTION"], "1")
        self.assertEqual(pe["PD_PREFIX_SNAPSHOT_BYTES"], "1073741824")
        self.assertEqual(de["PD_PREFIX_SNAPSHOT_BYTES"], "0")

    def test_invalid_role_and_missing_pool_id_fail_before_launch(self):
        self.assertNotEqual(self.launch("run_pd_http.sh", PD_ROLE="invalid").returncode, 0)
        self.env.pop("PD_CUDA_IPC_POOL_ID")
        result = self.launch("run_pd_http.sh", PD_ROLE="prefill", PD_PROFILE="b300-1p1d")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_pd_router_passes_distinct_workers(self):
        result = self.launch("run_pd_router.sh", ROUTER_BIN=str(self.recorder))
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout)["argv"]
        self.assertIn("--vllm-pd-disaggregation", argv)
        self.assertEqual(argv[argv.index("--prefill") + 1], "http://127.0.0.1:18300")
        self.assertEqual(argv[argv.index("--decode") + 1], "http://127.0.0.1:18400")

    def test_grpc_router_preserves_upstream_multiple_url_fix(self):
        result = self.launch("run_router.sh", ROUTER_BIN=str(self.recorder),
                             WORKER_URLS="grpc://127.0.0.1:50051,grpc://127.0.0.1:50052")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout)["argv"]
        self.assertEqual(argv[argv.index("--worker-urls") + 1:],
                         ["grpc://127.0.0.1:50051", "grpc://127.0.0.1:50052"])


if __name__ == "__main__":
    unittest.main()
