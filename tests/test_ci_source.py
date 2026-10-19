import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SourceValidationTests(unittest.TestCase):
    def test_application_sources_compile_without_importing_runtime_dependencies(self):
        source_files = [
            "app.py",
            "preprocess.py",
            "rl_environment.py",
            "dqn_agent.py",
            "train.py",
        ]

        for relative_path in source_files:
            with self.subTest(file=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                compile(source, relative_path, "exec")

    def test_required_api_routes_are_declared(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        routes = {
            decorator.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "route"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        }

        self.assertTrue({"/", "/health", "/api/skills"}.issubset(routes))


if __name__ == "__main__":
    unittest.main()
