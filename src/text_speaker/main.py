from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence


DEFAULT_CHARACTER = "feibi"
DEFAULT_TEXT = "你好，我是菲比。今天我们来测试 Genie 中文语音生成。"
DEFAULT_OUTPUT = Path("outputs/feibi_zh.wav")
PREDEFINED_CHARACTERS = ("feibi", "mika", "thirtyseven")
CHARACTER_ALIASES = {
    "feibi": "feibi",
    "菲比": "feibi",
    "mika": "mika",
    "misono mika": "mika",
    "圣园未花": "mika",
    "未花": "mika",
    "みその みか": "mika",
    "37": "thirtyseven",
    "thirtyseven": "thirtyseven",
}
HISTORY_DIR = Path("outputs/history")
HISTORY_FILE = HISTORY_DIR / "history.json"

_genie: ModuleType | None = None
_genie_lock = threading.RLock()
_loaded_characters: set[str] = set()


class SpeechGenerationCancelled(RuntimeError):
    """语音生成任务已被用户取消。"""


def available_characters() -> list[str]:
    """返回 Genie-TTS 当前示例支持选择的预置语音包。"""
    characters = set(PREDEFINED_CHARACTERS)
    model_root = Path("CharacterModels") / "v2ProPlus"
    if model_root.exists():
        characters.update(path.name for path in model_root.iterdir() if path.is_dir())

    ordered = [DEFAULT_CHARACTER]
    ordered.extend(sorted(character for character in characters if character != DEFAULT_CHARACTER))
    return ordered


def normalize_character(character: str) -> str:
    stripped_character = character.strip()
    return CHARACTER_ALIASES.get(
        stripped_character,
        CHARACTER_ALIASES.get(stripped_character.lower(), stripped_character.lower()),
    )


def validate_character(character: str) -> str:
    normalized_character = normalize_character(character)
    if normalized_character not in PREDEFINED_CHARACTERS:
        available = "、".join(PREDEFINED_CHARACTERS)
        raise ValueError(f"未知语音包：{character}。可用语音包：{available}")
    return normalized_character


def ensure_genie_data_exists() -> None:
    """在导入 genie_tts 前准备 GenieData，避免触发交互式下载确认。"""
    genie_data_dir = Path(os.getenv("GENIE_DATA_DIR", "./GenieData"))
    if genie_data_dir.exists():
        return

    if genie_data_dir.name != "GenieData":
        raise FileNotFoundError(
            f"GENIE_DATA_DIR 指向的目录不存在：{genie_data_dir}。"
            "自动下载仅支持默认 GenieData 目录，或 basename 为 GenieData 的目录。"
        )

    from huggingface_hub import snapshot_download

    genie_data_dir.parent.mkdir(parents=True, exist_ok=True)
    print("GenieData 不存在，正在从 HuggingFace 自动下载 Genie-TTS 基础资源...")
    snapshot_download(
        repo_id="High-Logic/Genie",
        repo_type="model",
        allow_patterns="GenieData/*",
        local_dir=str(genie_data_dir.parent),
        local_dir_use_symlinks=True,
    )


def _get_genie() -> ModuleType:
    global _genie
    if _genie is None:
        ensure_genie_data_exists()
        import genie_tts as genie

        _genie = genie
    return _genie


def ensure_character_loaded(character: str) -> ModuleType:
    """加载指定语音包；同一进程内同一语音包只加载一次。"""
    normalized_character = validate_character(character)
    with _genie_lock:
        genie = _get_genie()
        if normalized_character not in _loaded_characters:
            genie.load_predefined_character(normalized_character)
            _loaded_characters.add(normalized_character)
        return genie


def stop_speech() -> None:
    """停止当前 Genie-TTS 生成或播放任务。"""
    if _genie is None:
        return
    _genie.stop()


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        stop_speech()
        raise SpeechGenerationCancelled("已取消生成")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Genie-TTS 的菲比预置语音包生成中文语音。"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用默认命令行参数生成语音；不传任何参数时会打开 GUI 界面。",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="要合成的中文文本。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="生成音频的保存路径。",
    )
    parser.add_argument(
        "--character",
        default=DEFAULT_CHARACTER,
        help="Genie-TTS 预置角色名称，默认使用 feibi。",
    )
    parser.add_argument(
        "--genie-data-dir",
        type=Path,
        help="本地 GenieData 资源目录；会在导入 genie_tts 前写入 GENIE_DATA_DIR。",
    )
    parser.add_argument(
        "--download-roberta",
        action="store_true",
        help="当当前 genie-tts 版本提供 download_roberta_data() 时，下载中文推理可选 RoBERTa 资源。",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="生成后直接播放音频。",
    )
    return parser.parse_args(argv)


def generate_speech(
    *,
    text: str,
    output: Path,
    character: str = DEFAULT_CHARACTER,
    genie_data_dir: Path | None = None,
    download_roberta: bool = False,
    play: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    character = validate_character(character)

    if genie_data_dir is not None:
        os.environ["GENIE_DATA_DIR"] = str(genie_data_dir.expanduser().resolve())

    _raise_if_cancelled(should_cancel)
    genie = _get_genie()

    if download_roberta:
        download_roberta_data = getattr(genie, "download_roberta_data", None)
        if download_roberta_data is None:
            raise RuntimeError("当前安装的 genie-tts 版本未提供 download_roberta_data()。")
        download_roberta_data()

    output.parent.mkdir(parents=True, exist_ok=True)

    _raise_if_cancelled(should_cancel)
    genie = ensure_character_loaded(character)
    _raise_if_cancelled(should_cancel)

    genie.tts(
        character_name=character,
        text=text,
        play=play,
        save_path=str(output),
    )

    if play:
        genie.wait_for_playback_done()

    _raise_if_cancelled(should_cancel)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"语音生成失败，未写入有效音频文件：{output}")


def run_cli(args: argparse.Namespace) -> None:
    generate_speech(
        text=args.text,
        output=args.output,
        character=args.character,
        genie_data_dir=args.genie_data_dir,
        download_roberta=args.download_roberta,
        play=args.play,
    )
    print(f"语音生成完成：{args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        from text_speaker.gui import run_gui

        run_gui()
        return

    run_cli(parse_args(argv))


if __name__ == "__main__":
    main()
