from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from .main import (
    DEFAULT_CHARACTER,
    DEFAULT_TEXT,
    HISTORY_DIR,
    HISTORY_FILE,
    available_character_display_names,
    character_display_name,
    generate_speech,
    normalize_character,
    SpeechGenerationCancelled,
    stop_speech,
)


@dataclass(frozen=True)
class HistoryItem:
    text: str
    character: str
    audio_path: str
    created_at: str
    deleted_at: str | None = None


class GenieTtsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Genie 中文语音生成")
        self.root.minsize(720, 560)

        self.history: list[HistoryItem] = self._load_history()
        self.is_generating = False
        self.current_task_id = 0
        self.cancelled_task_ids: set[int] = set()
        self.worker: threading.Thread | None = None
        self.is_closing = False
        self.last_generated: HistoryItem | None = self._latest_active_history_item()

        self.character_var = tk.StringVar(value=character_display_name(DEFAULT_CHARACTER))
        self.status_var = tk.StringVar(value="就绪")
        self.generate_button_text = tk.StringVar(value="生成 (Enter)")

        self._build_style()
        self._build_ui()
        self._render_history()
        self._refresh_primary_button()
        self.root.bind("<Return>", self._handle_enter)
        self.root.bind("<Escape>", self._handle_escape)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self) -> None:
        style = ttk.Style()
        style.configure("Section.TFrame", padding=16)
        style.configure("Muted.TLabel", foreground="#5f6368")
        style.configure("Primary.TButton", padding=(16, 0))
        style.configure("History.TFrame", padding=(12, 10))

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        input_section = ttk.Frame(container, style="Section.TFrame")
        input_section.grid(row=0, column=0, sticky="ew")
        input_section.columnconfigure(0, weight=1)
        input_section.rowconfigure(0, weight=1)

        self.text_input = tk.Text(
            input_section,
            height=7,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=10,
            font=("Microsoft YaHei UI", 11),
            undo=True,
        )
        self.text_input.insert("1.0", DEFAULT_TEXT)
        self.text_input.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.text_input.bind("<Return>", self._handle_enter)
        self.text_input.bind("<Shift-Return>", self._insert_newline)
        self.text_input.bind("<KeyRelease>", self._on_input_changed)

        control_bar = ttk.Frame(input_section)
        control_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        control_bar.columnconfigure(1, weight=1)

        self.character_select = ttk.Combobox(
            control_bar,
            textvariable=self.character_var,
            values=available_character_display_names(),
            state="readonly",
            width=30,
        )
        self.character_select.grid(row=0, column=0, sticky="w")
        self.character_select.bind("<<ComboboxSelected>>", self._on_input_changed)

        self.generate_button = ttk.Button(
            control_bar,
            textvariable=self.generate_button_text,
            command=self._generate_or_cancel,
            style="Primary.TButton",
        )
        self.generate_button.grid(row=0, column=2, sticky="e")

        self.status_label = ttk.Label(
            input_section,
            textvariable=self.status_var,
            style="Muted.TLabel",
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        history_section = ttk.Frame(container, style="Section.TFrame")
        history_section.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        history_section.columnconfigure(0, weight=1)
        history_section.rowconfigure(1, weight=1)

        history_header = ttk.Frame(history_section)
        history_header.grid(row=0, column=0, columnspan=2, sticky="ew")
        history_header.columnconfigure(0, weight=1)

        ttk.Label(history_header, text="历史记录").grid(row=0, column=0, sticky="w")

        self.clear_history_button = ttk.Button(
            history_header,
            text="清理",
            command=self._clear_visible_history,
        )
        self.open_history_folder_button = ttk.Button(
            history_header,
            text="打开文件夹",
            command=self._open_history_folder,
        )
        self.open_history_folder_button.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self.clear_history_button.grid(row=0, column=2, sticky="e")

        self.history_canvas = tk.Canvas(
            history_section,
            borderwidth=0,
            highlightthickness=0,
        )
        self.history_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        scrollbar = ttk.Scrollbar(
            history_section,
            orient="vertical",
            command=self.history_canvas.yview,
        )
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.history_canvas.configure(yscrollcommand=scrollbar.set)

        self.history_frame = ttk.Frame(self.history_canvas)
        self.history_window = self.history_canvas.create_window(
            (0, 0),
            window=self.history_frame,
            anchor="nw",
        )
        self.history_frame.bind("<Configure>", self._update_history_scroll_region)
        self.history_canvas.bind("<Configure>", self._resize_history_frame)

    def _handle_enter(self, event: tk.Event) -> str:
        self._generate_cancel_or_play()
        return "break"

    def _handle_escape(self, event: tk.Event) -> str:
        if self.is_generating:
            self._cancel_generation()
            return "break"
        return "continue"

    def _insert_newline(self, event: tk.Event) -> str:
        self.text_input.insert("insert", "\n")
        return "break"

    def _generate_or_cancel(self) -> None:
        self._generate_cancel_or_play()

    def _generate_cancel_or_play(self) -> None:
        if self.is_generating:
            self._cancel_generation()
            return

        text = self._current_text()
        character = normalize_character(self.character_var.get())
        if (
            self.last_generated is not None
            and self.last_generated.text == text
            and self.last_generated.character == character
        ):
            self._play_audio(Path(self.last_generated.audio_path))
            return

        self._start_generation()

    def _matches_last_generated(self) -> bool:
        if self.last_generated is None:
            return False
        return (
            self.last_generated.text == self._current_text()
            and self.last_generated.character == normalize_character(self.character_var.get())
            and Path(self.last_generated.audio_path).exists()
        )

    def _refresh_primary_button(self) -> None:
        if self.is_generating:
            self.generate_button_text.set("取消 (Enter / Esc)")
        elif self._matches_last_generated():
            self.generate_button_text.set("播放 (Enter)")
        else:
            self.generate_button_text.set("生成 (Enter)")

    def _on_input_changed(self, event: tk.Event) -> None:
        self._refresh_primary_button()

    def _current_text(self) -> str:
        return self.text_input.get("1.0", "end-1c").strip()

    def _start_generation(self) -> None:
        text = self._current_text()
        if not text:
            self.status_var.set("请输入需要生成的文本")
            return

        character = normalize_character(self.character_var.get())
        output_path = self._next_output_path(character)
        self.current_task_id += 1
        task_id = self.current_task_id
        self.cancelled_task_ids.discard(task_id)
        self._set_generating(True)
        self.status_var.set("准备生成语音...")

        self.worker = threading.Thread(
            target=self._run_generation,
            args=(task_id, text, character, output_path),
            daemon=True,
        )
        self.worker.start()

    def _run_generation(
        self,
        task_id: int,
        text: str,
        character: str,
        output_path: Path,
    ) -> None:
        try:
            generate_speech(
                text=text,
                output=output_path,
                character=character,
                play=False,
                should_cancel=lambda: task_id in self.cancelled_task_ids,
                status_callback=lambda message: self._update_generation_status(task_id, message),
            )
        except SpeechGenerationCancelled:
            self._call_on_ui_thread(self._on_generation_cancelled, task_id)
            return
        except Exception as exc:
            self._call_on_ui_thread(self._on_generation_failed, task_id, str(exc))
            return

        item = HistoryItem(
            text=text,
            character=character,
            audio_path=str(output_path),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._call_on_ui_thread(self._on_generation_succeeded, task_id, item)

    def _update_generation_status(self, task_id: int, message: str) -> None:
        self._call_on_ui_thread(self._set_generation_status, task_id, message)

    def _call_on_ui_thread(self, callback: Callable[..., None], *args: object) -> None:
        if self.is_closing:
            return
        try:
            self.root.after(0, callback, *args)
        except tk.TclError:
            return

    def _set_generation_status(self, task_id: int, message: str) -> None:
        if self.is_generating and task_id == self.current_task_id:
            self.status_var.set(message)

    def _cancel_generation(self) -> None:
        if not self.is_generating:
            return
        self.cancelled_task_ids.add(self.current_task_id)
        self.status_var.set("正在取消...")
        self.generate_button.state(["disabled"])
        threading.Thread(target=self._stop_current_generation, daemon=True).start()

    def _stop_current_generation(self) -> None:
        task_id = self.current_task_id
        try:
            stop_speech()
        finally:
            self._call_on_ui_thread(self._on_generation_cancelled, task_id)

    def _on_generation_succeeded(self, task_id: int, item: HistoryItem) -> None:
        if task_id in self.cancelled_task_ids:
            return
        if not self.is_generating or task_id != self.current_task_id:
            return
        self._set_generating(False)
        self.last_generated = item
        self.history.insert(0, item)
        self._save_history()
        self._render_history()
        self.status_var.set(f"生成完成：{Path(item.audio_path).name}")
        self._refresh_primary_button()

    def _on_generation_failed(self, task_id: int, message: str) -> None:
        if task_id in self.cancelled_task_ids:
            return
        if not self.is_generating or task_id != self.current_task_id:
            return
        self._set_generating(False)
        self.status_var.set(self._format_error_status(message))
        messagebox.showerror("生成失败", message)

    def _on_generation_cancelled(self, task_id: int) -> None:
        if task_id != self.current_task_id:
            return
        if not self.is_generating:
            return
        self._set_generating(False)
        self.status_var.set("已取消生成")

    def _set_generating(self, is_generating: bool) -> None:
        self.is_generating = is_generating
        self._refresh_primary_button()
        self.generate_button.state(["!disabled"])
        self.text_input.configure(state="disabled" if is_generating else "normal")
        self.character_select.configure(state="disabled" if is_generating else "readonly")

    def _next_output_path(self, character: str) -> Path:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return HISTORY_DIR / f"{timestamp}_{character}.wav"

    def _play_audio(self, audio_path: Path) -> None:
        if not audio_path.exists():
            self.status_var.set("音频文件不存在，无法播放")
            return

        try:
            if sys.platform.startswith("win"):
                import winsound

                winsound.PlaySound(
                    str(audio_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["afplay", str(audio_path)])
            else:
                subprocess.Popen(["xdg-open", str(audio_path)])
        except Exception as exc:
            messagebox.showerror("播放失败", str(exc))
            return

        self.status_var.set(f"正在播放：{audio_path.name}")

    def _stop_audio_playback(self) -> None:
        if not sys.platform.startswith("win"):
            return

        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:
            return

    def _render_history(self) -> None:
        for child in self.history_frame.winfo_children():
            child.destroy()

        visible_history = self._active_history_items()
        self.clear_history_button.state(["!disabled"] if visible_history else ["disabled"])

        if not visible_history:
            ttk.Label(
                self.history_frame,
                text="暂无历史记录",
                style="Muted.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=10)
            return

        self.history_frame.columnconfigure(0, weight=1)
        for row, item in enumerate(visible_history):
            row_frame = ttk.Frame(self.history_frame, style="History.TFrame")
            row_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            row_frame.columnconfigure(0, weight=1)

            text_label = ttk.Label(
                row_frame,
                text=self._format_history_text(item),
                wraplength=560,
                justify="left",
            )
            text_label.grid(row=0, column=0, sticky="ew")

            delete_button = ttk.Button(
                row_frame,
                text="删除",
                command=lambda history_item=item: self._mark_history_item_deleted(history_item),
            )
            delete_button.grid(row=0, column=1, sticky="e", padx=(12, 0))

            open_button = ttk.Button(
                row_frame,
                text="打开",
                command=lambda path=item.audio_path: self._open_audio_location(Path(path)),
            )
            open_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

            play_button = ttk.Button(
                row_frame,
                text="播放",
                command=lambda path=item.audio_path: self._play_audio(Path(path)),
            )
            play_button.grid(row=0, column=3, sticky="e", padx=(8, 0))

    def _open_history_folder(self) -> None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        self._open_folder(HISTORY_DIR)

    def _open_audio_location(self, audio_path: Path) -> None:
        if not audio_path.exists():
            self.status_var.set("音频文件不存在，无法打开")
            return

        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(audio_path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(audio_path)])
            else:
                self._open_folder(audio_path.parent)
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))
            return

        self.status_var.set(f"已打开：{audio_path.name}")

    def _open_folder(self, folder_path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(folder_path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder_path)])
            else:
                subprocess.Popen(["xdg-open", str(folder_path)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))
            return

        self.status_var.set(f"已打开文件夹：{folder_path}")

    def _active_history_items(self) -> list[HistoryItem]:
        return [item for item in self.history if item.deleted_at is None]

    def _latest_active_history_item(self) -> HistoryItem | None:
        for item in self.history:
            if item.deleted_at is None:
                return item
        return None

    def _mark_history_item_deleted(self, item: HistoryItem) -> None:
        deleted_at = datetime.now().isoformat(timespec="seconds")
        updated_history: list[HistoryItem] = []
        for current_item in self.history:
            if current_item is item:
                updated_history.append(replace(current_item, deleted_at=deleted_at))
            else:
                updated_history.append(current_item)

        self.history = updated_history
        self._refresh_last_generated_after_history_change()
        self._save_history()
        self._render_history()
        self.status_var.set("已从历史记录中删除")

    def _clear_visible_history(self) -> None:
        deleted_at = datetime.now().isoformat(timespec="seconds")
        self.history = [
            replace(item, deleted_at=deleted_at) if item.deleted_at is None else item
            for item in self.history
        ]
        self._refresh_last_generated_after_history_change()
        self._save_history()
        self._render_history()
        self.status_var.set("已清理历史记录")

    def _refresh_last_generated_after_history_change(self) -> None:
        if self.last_generated is not None and self.last_generated.deleted_at is None:
            for item in self._active_history_items():
                if (
                    item.audio_path == self.last_generated.audio_path
                    and item.created_at == self.last_generated.created_at
                ):
                    self.last_generated = item
                    self._refresh_primary_button()
                    return
        self.last_generated = self._latest_active_history_item()
        self._refresh_primary_button()

    def _format_history_text(self, item: HistoryItem) -> str:
        preview = item.text.replace("\n", " ")
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        return f"[{character_display_name(item.character)}] {preview}"

    def _format_error_status(self, message: str) -> str:
        one_line_message = " ".join(message.split())
        if len(one_line_message) > 80:
            one_line_message = f"{one_line_message[:77]}..."
        return one_line_message

    def _load_history(self) -> list[HistoryItem]:
        if not HISTORY_FILE.exists():
            return []

        try:
            raw_items = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        history: list[HistoryItem] = []
        for raw_item in raw_items:
            try:
                item = HistoryItem(
                    text=str(raw_item["text"]),
                    character=normalize_character(str(raw_item["character"])),
                    audio_path=str(raw_item["audio_path"]),
                    created_at=str(raw_item["created_at"]),
                    deleted_at=(
                        str(raw_item["deleted_at"])
                        if raw_item.get("deleted_at") is not None
                        else None
                    ),
                )
            except KeyError:
                continue
            if Path(item.audio_path).exists():
                history.append(item)
        return history

    def _save_history(self) -> None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps([asdict(item) for item in self.history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _update_history_scroll_region(self, event: tk.Event) -> None:
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def _resize_history_frame(self, event: tk.Event) -> None:
        self.history_canvas.itemconfigure(self.history_window, width=event.width)

    def _on_close(self) -> None:
        if self.is_closing:
            return
        self.is_closing = True
        if self.is_generating:
            self.cancelled_task_ids.add(self.current_task_id)
            stop_speech()
        self._stop_audio_playback()

        try:
            self.root.quit()
            self.root.destroy()
        finally:
            if sys.platform.startswith("win") and getattr(sys, "frozen", False):
                os._exit(0)


def run_gui() -> None:
    root = tk.Tk()
    app = GenieTtsApp(root)
    root.mainloop()
