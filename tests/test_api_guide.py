"""The documented examples must run against the actual public interfaces."""

from pathlib import Path
import re


def test_python_api_guide_examples_execute():
    guide = Path(__file__).resolve().parents[1] / "PYTHON_API.md"
    namespace = {}
    for index, block in enumerate(
        re.findall(r"```python\n(.*?)```", guide.read_text(), re.S)
    ):
        exec(compile(block, f"PYTHON_API.md:block-{index + 1}", "exec"), namespace)
