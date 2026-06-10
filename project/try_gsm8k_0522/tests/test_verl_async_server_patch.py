import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASYNC_SERVER = ROOT / "agentflow" / "verl" / "async_server.py"
VLLM_INSTRUMENTATION = ROOT / "agentflow" / "instrumentation" / "vllm.py"


class TestVerlAsyncServerPatch(unittest.TestCase):
    def test_uses_verl_http_server_without_overriding_chat_completion(self):
        source = ASYNC_SERVER.read_text()
        tree = ast.parse(source)

        imports_vllm_http_server = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "verl.workers.rollout.vllm_rollout.vllm_async_server"
            and any(alias.name == "vLLMHttpServer" for alias in node.names)
            for node in tree.body
        )
        self.assertTrue(imports_vllm_http_server)
        self.assertNotIn("AsyncvLLMServer", source)

        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        patched_server = next(node for node in classes if node.name == "PatchedvLLMServer")

        base_names = {getattr(base, "id", None) for base in patched_server.bases}
        self.assertIn("vLLMHttpServer", base_names)

        method_names = {node.name for node in patched_server.body if isinstance(node, ast.AsyncFunctionDef)}
        self.assertIn("run_server", method_names)
        self.assertNotIn("chat_completion", method_names)

    def test_run_server_sets_vllm_app_state_server(self):
        source = ASYNC_SERVER.read_text()
        tree = ast.parse(source)
        patched_server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PatchedvLLMServer")
        run_server = next(
            node
            for node in patched_server.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_server"
        )

        assigns_state_server = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "server"
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "state"
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "app"
                for target in node.targets
            )
            for node in ast.walk(run_server)
        )
        calls_super_run_server = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_server"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            for node in ast.walk(run_server)
        )

        self.assertTrue(assigns_state_server)
        self.assertFalse(calls_super_run_server)

    def test_vllm_instrumentation_does_not_require_protocol_at_import_time(self):
        source = VLLM_INSTRUMENTATION.read_text()
        tree = ast.parse(source)

        top_level_protocol_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "vllm.entrypoints.openai.protocol"
        ]

        self.assertEqual(top_level_protocol_imports, [])


if __name__ == "__main__":
    unittest.main()
