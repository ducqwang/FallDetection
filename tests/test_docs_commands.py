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


TRACK_SCRIPTS = ("scripts/make_tracks.py", "scripts/check_tracks_quality.py")


def test_track_commands_in_docs_use_real_flags():
    """Lệnh make_tracks / check_tracks_quality trong HANDOFF, README và notebook 02 chỉ dùng cờ có thật."""
    helps = {script: subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True).stdout
             for script in TRACK_SCRIPTS}
    texts = [(ROOT / name).read_text(encoding="utf-8") for name in
             ("docs/HANDOFF_TV1.md", "README.md", "notebooks/02_m1_baseline.ipynb", "notebooks/tv3_05_m3.ipynb")]
    commands = [line for text in texts for line in text.splitlines() if any(s in line for s in TRACK_SCRIPTS)
                and "python " in line]
    assert len(commands) >= 6
    for command in commands:
        script = next(s for s in TRACK_SCRIPTS if s in command)
        for flag in re.findall(r"(?<=\s)--[a-z][a-z-]*", command.split(script, 1)[1]):
            assert flag in helps[script], f"{flag} không có trong {script}: {command}"
