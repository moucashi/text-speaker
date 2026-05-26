from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence


DEFAULT_CHARACTER = "feibi"
DEFAULT_TEXT = "你好，我是菲比。今天我们来测试 Genie 中文语音生成。"
DEFAULT_OUTPUT = Path("outputs/feibi_zh.wav")
MODELS_DIR = Path("models")
DEFAULT_GENIE_DATA_DIR = MODELS_DIR / "GenieData"
CHARACTER_MODELS_DIR = MODELS_DIR / "CharacterModels"
CHARACTER_MODEL_VERSION = "v2ProPlus"
PREDEFINED_CHARACTERS = ("feibi", "mika", "thirtyseven")
CHARACTER_LANGUAGES = {
    "feibi": "Chinese",
    "mika": "Japanese",
    "thirtyseven": "English",
}
CHARACTER_DISPLAY_NAMES = {
    "feibi": "[中文] 菲比 (feibi)",
    "mika": "[日语] 圣园未花 / Misono Mika (mika)",
    "thirtyseven": "[英语] 37 / Thirty Seven (thirtyseven)",
}
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
REQUIRED_GENIE_DATA_FILES = ("speaker_encoder.onnx",)
REQUIRED_CHARACTER_PATHS = (
    "prompt_wav.json",
    "prompt_wav",
    "tts_models/t2s_encoder_fp32.bin",
    "tts_models/t2s_encoder_fp32.onnx",
    "tts_models/t2s_first_stage_decoder_fp32.onnx",
    "tts_models/t2s_shared_fp16.bin",
    "tts_models/t2s_stage_decoder_fp32.onnx",
    "tts_models/vits_fp16.bin",
    "tts_models/vits_fp32.onnx",
    "tts_models/prompt_encoder_fp16.bin",
    "tts_models/prompt_encoder_fp32.onnx",
)
HISTORY_DIR = Path("outputs/history")
HISTORY_FILE = HISTORY_DIR / "history.json"

_genie: ModuleType | None = None
_genie_lock = threading.RLock()
_loaded_characters: set[str] = set()


@dataclass(frozen=True)
class StatusUpdate:
    message: str
    progress: float | None = None
    busy: bool = False


StatusPayload = str | StatusUpdate
StatusCallback = Callable[[StatusPayload], None]


def _ensure_standard_streams() -> None:
    """为无控制台的 Windows EXE 提供静默标准流，避免第三方库写日志时报错。"""
    missing_stream = False
    if sys.stdin is None:
        sys.stdin = sys.__stdin__ = open(os.devnull, "r", encoding="utf-8")
        missing_stream = True
    if sys.stdout is None:
        sys.stdout = sys.__stdout__ = open(os.devnull, "w", encoding="utf-8")
        missing_stream = True
    if sys.stderr is None:
        sys.stderr = sys.__stderr__ = open(os.devnull, "w", encoding="utf-8")
        missing_stream = True
    if missing_stream:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


_ensure_standard_streams()


class SpeechGenerationCancelled(RuntimeError):
    """语音生成任务已被用户取消。"""


def available_characters() -> list[str]:
    """返回 Genie-TTS 当前示例支持选择的预置语音包。"""
    characters = set(PREDEFINED_CHARACTERS)
    model_root = CHARACTER_MODELS_DIR / CHARACTER_MODEL_VERSION
    if model_root.exists():
        characters.update(path.name for path in model_root.iterdir() if path.is_dir())

    ordered = [DEFAULT_CHARACTER]
    ordered.extend(sorted(character for character in characters if character != DEFAULT_CHARACTER))
    return ordered


def character_display_name(character: str) -> str:
    """返回语音包在界面中使用的展示名称。"""
    normalized_character = normalize_character(character)
    return CHARACTER_DISPLAY_NAMES.get(normalized_character, normalized_character)


def available_character_display_names() -> list[str]:
    """返回 Genie-TTS 当前示例支持选择的预置语音包展示名。"""
    return [character_display_name(character) for character in available_characters()]


def normalize_character(character: str) -> str:
    stripped_character = character.strip()
    if stripped_character in CHARACTER_DISPLAY_NAMES.values():
        return next(
            character_id
            for character_id, display_name in CHARACTER_DISPLAY_NAMES.items()
            if display_name == stripped_character
        )

    lowered_character = stripped_character.lower()
    return CHARACTER_ALIASES.get(
        stripped_character,
        CHARACTER_ALIASES.get(lowered_character, lowered_character),
    )


def validate_character(character: str) -> str:
    normalized_character = normalize_character(character)
    if normalized_character not in PREDEFINED_CHARACTERS:
        available = "、".join(available_character_display_names())
        raise ValueError(f"未知语音包：{character}。可用语音包：{available}")
    return normalized_character


def _emit_status(
    status_callback: StatusCallback | None,
    message: str,
    *,
    progress: float | None = None,
    busy: bool = False,
) -> None:
    if status_callback is not None:
        if progress is None and not busy:
            status_callback(message)
        else:
            status_callback(StatusUpdate(message=message, progress=progress, busy=busy))


def _download_tqdm_class(
    status_callback: StatusCallback | None,
    message_prefix: str,
):
    if status_callback is None:
        return None

    from tqdm.auto import tqdm

    class GuiDownloadProgress(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._gui_last_percent = -1
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            total = self.total
            if total:
                percent = min(100.0, max(0.0, self.n / total * 100))
                whole_percent = int(percent)
                if whole_percent != self._gui_last_percent:
                    self._gui_last_percent = whole_percent
                    _emit_status(
                        status_callback,
                        f"{message_prefix}：{whole_percent}%",
                        progress=percent,
                    )
            else:
                _emit_status(status_callback, message_prefix, busy=True)
            return result

        def close(self) -> None:
            total = self.total
            if total and self.n >= total:
                _emit_status(status_callback, f"{message_prefix}：100%", progress=100.0)
            super().close()

    return GuiDownloadProgress


def _raise_stage_error(
    status_callback: StatusCallback | None,
    stage: str,
    exc: Exception,
) -> None:
    message = f"{stage}失败：{exc}"
    _emit_status(status_callback, message)
    raise RuntimeError(message) from exc


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _missing_paths(directory: Path, required_paths: Sequence[str]) -> list[Path]:
    return [
        directory / relative_path
        for relative_path in required_paths
        if not (directory / relative_path).exists()
    ]


def _remove_incomplete_model_dir(directory: Path, status_callback: StatusCallback | None) -> None:
    if not directory.exists():
        return
    if not _path_is_relative_to(directory, MODELS_DIR):
        raise RuntimeError(f"拒绝自动清理模型目录外的路径：{directory}")

    _emit_status(status_callback, f"正在清理未完成的模型下载：{directory}", busy=True)
    shutil.rmtree(directory)


def _validate_required_paths(
    directory: Path,
    required_paths: Sequence[str],
    resource_name: str,
) -> None:
    missing_paths = _missing_paths(directory, required_paths)
    if missing_paths:
        formatted_paths = "、".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"{resource_name} 不完整，缺少：{formatted_paths}")


def ensure_genie_data_exists(status_callback: StatusCallback | None = None) -> None:
    """在导入 genie_tts 前准备 GenieData，避免触发交互式下载确认。"""
    genie_data_dir = Path(os.getenv("GENIE_DATA_DIR", str(DEFAULT_GENIE_DATA_DIR)))
    os.environ["GENIE_DATA_DIR"] = str(genie_data_dir)
    if genie_data_dir.exists():
        missing_paths = _missing_paths(genie_data_dir, REQUIRED_GENIE_DATA_FILES)
        if not missing_paths:
            return
        if not _path_is_relative_to(genie_data_dir, MODELS_DIR):
            formatted_paths = "、".join(str(path) for path in missing_paths)
            message = f"Genie-TTS 基础资源目录不完整，缺少：{formatted_paths}"
            _emit_status(status_callback, message)
            raise FileNotFoundError(message)
        _emit_status(status_callback, "检测到未完成的 Genie-TTS 基础资源下载，正在重新下载...", busy=True)
        _remove_incomplete_model_dir(genie_data_dir, status_callback)

    if genie_data_dir.name != "GenieData":
        message = (
            f"Genie-TTS 基础资源目录不存在：{genie_data_dir}。"
            "自动下载仅支持默认 GenieData 目录，或 basename 为 GenieData 的目录。"
        )
        _emit_status(status_callback, message)
        raise FileNotFoundError(message)

    from huggingface_hub import snapshot_download

    genie_data_dir.parent.mkdir(parents=True, exist_ok=True)
    _emit_status(status_callback, "正在下载 Genie-TTS 基础资源...", progress=0.0)
    print("GenieData 不存在，正在从 HuggingFace 自动下载 Genie-TTS 基础资源...")
    try:
        snapshot_download(
            repo_id="High-Logic/Genie",
            repo_type="model",
            allow_patterns="GenieData/*",
            local_dir=str(genie_data_dir.parent),
            local_dir_use_symlinks=True,
            tqdm_class=_download_tqdm_class(status_callback, "正在下载 Genie-TTS 基础资源"),
        )
    except Exception as exc:
        _raise_stage_error(status_callback, "下载 Genie-TTS 基础资源", exc)
    try:
        _validate_required_paths(
            genie_data_dir,
            REQUIRED_GENIE_DATA_FILES,
            "Genie-TTS 基础资源",
        )
    except Exception as exc:
        _raise_stage_error(status_callback, "校验 Genie-TTS 基础资源", exc)


def _get_genie(status_callback: StatusCallback | None = None) -> ModuleType:
    global _genie
    with _genie_lock:
        if _genie is None:
            ensure_genie_data_exists(status_callback)
            _emit_status(status_callback, "正在初始化 Genie-TTS...", busy=True)
            try:
                import genie_tts as genie
            except Exception as exc:
                _raise_stage_error(status_callback, "初始化 Genie-TTS", exc)

            _genie = genie
        return _genie


def _download_predefined_character(
    character: str,
    status_callback: StatusCallback | None = None,
) -> Path:
    character_dir = CHARACTER_MODELS_DIR / CHARACTER_MODEL_VERSION / character
    if character_dir.exists():
        missing_paths = _missing_paths(character_dir, REQUIRED_CHARACTER_PATHS)
        if not missing_paths:
            return character_dir
        display_name = character_display_name(character)
        _emit_status(status_callback, f"检测到未完成的 {display_name} 语音包下载，正在重新下载...", busy=True)
        _remove_incomplete_model_dir(character_dir, status_callback)

    from huggingface_hub import snapshot_download

    CHARACTER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    display_name = character_display_name(character)
    _emit_status(status_callback, f"正在下载 {display_name} 语音包...", progress=0.0)
    try:
        snapshot_download(
            repo_id="High-Logic/Genie",
            repo_type="model",
            allow_patterns=f"CharacterModels/{CHARACTER_MODEL_VERSION}/{character}/*",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=True,
            tqdm_class=_download_tqdm_class(status_callback, f"正在下载 {display_name} 语音包"),
        )
    except Exception as exc:
        _raise_stage_error(status_callback, f"下载 {display_name} 语音包", exc)

    try:
        _validate_required_paths(
            character_dir,
            REQUIRED_CHARACTER_PATHS,
            f"{display_name} 语音包",
        )
    except Exception as exc:
        _raise_stage_error(status_callback, f"校验 {display_name} 语音包", exc)
    return character_dir


def _load_predefined_character_from_dir(
    genie: ModuleType,
    character: str,
    character_dir: Path,
) -> None:
    prompt_config_path = character_dir / "prompt_wav.json"
    with prompt_config_path.open("r", encoding="utf-8") as file:
        prompt_wav_dict = json.load(file)

    prompt_config = prompt_wav_dict["Normal"]
    genie.load_character(
        character_name=character,
        onnx_model_dir=character_dir / "tts_models",
        language=CHARACTER_LANGUAGES[character],
    )
    genie.set_reference_audio(
        character_name=character,
        audio_path=character_dir / "prompt_wav" / prompt_config["wav"],
        audio_text=prompt_config["text"],
        language=CHARACTER_LANGUAGES[character],
    )


def ensure_character_loaded(
    character: str,
    status_callback: StatusCallback | None = None,
) -> ModuleType:
    """加载指定语音包；同一进程内同一语音包只加载一次。"""
    normalized_character = validate_character(character)
    with _genie_lock:
        genie = _get_genie(status_callback)
        if normalized_character not in _loaded_characters:
            display_name = character_display_name(normalized_character)
            character_dir = (
                CHARACTER_MODELS_DIR / CHARACTER_MODEL_VERSION / normalized_character
            )
            if character_dir.exists():
                _emit_status(status_callback, f"正在加载 {display_name} 语音包...", busy=True)
            else:
                _emit_status(status_callback, f"正在下载并加载 {display_name} 语音包...", busy=True)
            character_dir = _download_predefined_character(
                normalized_character,
                status_callback,
            )
            _emit_status(status_callback, f"正在加载 {display_name} 语音包...", busy=True)
            try:
                _load_predefined_character_from_dir(
                    genie,
                    normalized_character,
                    character_dir,
                )
            except Exception as exc:
                _raise_stage_error(status_callback, f"加载 {display_name} 语音包", exc)
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
    status_callback: StatusCallback | None = None,
) -> None:
    character = validate_character(character)

    if genie_data_dir is not None:
        os.environ["GENIE_DATA_DIR"] = str(genie_data_dir.expanduser().resolve())

    _raise_if_cancelled(should_cancel)
    genie = _get_genie(status_callback)

    if download_roberta:
        _emit_status(status_callback, "正在下载 RoBERTa 文本特征资源...", busy=True)
        download_roberta_data = getattr(genie, "download_roberta_data", None)
        if download_roberta_data is None:
            message = "下载 RoBERTa 文本特征资源失败：当前安装的 genie-tts 版本未提供 download_roberta_data()。"
            _emit_status(status_callback, message)
            raise RuntimeError(message)
        try:
            download_roberta_data()
        except Exception as exc:
            _raise_stage_error(status_callback, "下载 RoBERTa 文本特征资源", exc)

    output.parent.mkdir(parents=True, exist_ok=True)

    _raise_if_cancelled(should_cancel)
    genie = ensure_character_loaded(character, status_callback)
    _raise_if_cancelled(should_cancel)

    _emit_status(status_callback, "正在生成语音...", busy=True)
    try:
        genie.tts(
            character_name=character,
            text=text,
            play=play,
            save_path=str(output),
        )
    except Exception as exc:
        _raise_stage_error(status_callback, "生成语音", exc)

    if play:
        try:
            genie.wait_for_playback_done()
        except Exception as exc:
            _raise_stage_error(status_callback, "等待语音播放完成", exc)

    _raise_if_cancelled(should_cancel)
    if not output.exists() or output.stat().st_size == 0:
        message = f"生成语音失败：未写入有效音频文件：{output}"
        _emit_status(status_callback, message)
        raise RuntimeError(message)


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
