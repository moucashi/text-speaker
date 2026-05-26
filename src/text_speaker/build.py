from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT_DIR / "src" / "text_speaker" / "main.py"
SRC_DIR = ROOT_DIR / "src"
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build" / "pyinstaller"
PYINSTALLER_GENIE_DATA_DIR = BUILD_DIR / "analysis" / "GenieData"
GENIE_RUNTIME_DATA_PACKAGES = (
    "g2pM",
    "jieba_fast",
    "pypinyin",
    "pyopenjtalk",
)


def prepare_pyinstaller_genie_data_dir() -> None:
    """提供 PyInstaller 分析阶段使用的最小 GenieData 目录，跳过第三方包交互确认。"""
    PYINSTALLER_GENIE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PYINSTALLER_GENIE_DATA_DIR / "chinese-hubert-base").mkdir(exist_ok=True)
    (PYINSTALLER_GENIE_DATA_DIR / "speaker_encoder.onnx").touch()


def run_pyinstaller(*, name: str, windowed: bool) -> None:
    release_dir = DIST_DIR / name
    try:
        if release_dir.exists():
            shutil.rmtree(release_dir)
    except PermissionError as exc:
        raise RuntimeError(
            f"无法清理 {release_dir}，请先关闭正在运行的 {name}.exe 后重试。"
        ) from exc

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        name,
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / name),
        "--specpath",
        str(BUILD_DIR / "spec"),
        "--paths",
        str(SRC_DIR),
        "--collect-all",
        "genie_tts",
    ]
    for package in GENIE_RUNTIME_DATA_PACKAGES:
        command.extend(["--collect-data", package])

    if windowed:
        command.append("--windowed")
    command.append(str(ENTRYPOINT))

    env = os.environ.copy()
    env["GENIE_DATA_DIR"] = str(PYINSTALLER_GENIE_DATA_DIR)

    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def main() -> None:
    if not ENTRYPOINT.exists():
        raise FileNotFoundError(f"未找到入口文件：{ENTRYPOINT}")

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    prepare_pyinstaller_genie_data_dir()
    run_pyinstaller(name="text-speaker", windowed=True)
    run_pyinstaller(name="text-speaker-console", windowed=False)

    print("EXE 打包完成：")
    print(f"- {DIST_DIR / 'text-speaker' / 'text-speaker.exe'}")
    print(f"- {DIST_DIR / 'text-speaker-console' / 'text-speaker-console.exe'}")
