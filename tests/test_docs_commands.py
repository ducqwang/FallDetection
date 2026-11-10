"""Mọi cờ trong lệnh mẫu của docs/EVAL_PROTOCOL.md (mục Quy trình M1) phải có trong argparse thật của script."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {"scripts/tune_m1.py": ["scripts/tune_m1.py"], "scripts/evaluate.py": ["scripts/evaluate.py"],
           "src.models.m1_rule": ["-m", "src.models.m1_rule"]}


def m1_commands() -> list[str]:
    text = (ROOT / "docs" / "EVAL_PROTOCOL.md").read_text(encoding="utf-8")
    section = text[text.index("### Quy trình M1"):]
    return [line for line in section.splitlines() if line.startswith("python ")]


def test_every_flag_in_m1_procedure_exists_in_argparse():
    commands = m1_commands()
    assert len(commands) >= 8
    helps = {key: subprocess.run([sys.executable, *args, "--help"], cwd=ROOT, capture_output=True, text=True).stdout
             for key, args in SCRIPTS.items()}
    for command in commands:
        script = next(key for key in SCRIPTS if key in command)
        for flag in re.findall(r"(?<=\s)--[a-z][a-z-]*", command):
            assert flag in helps[script], f"{flag} không có trong {script}: {command}"
