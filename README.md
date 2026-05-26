# Text Speaker

这是一个基于 [High-Logic/Genie-TTS](https://github.com/High-Logic/Genie-TTS) 的语音生成示例项目，默认使用菲比（`feibi`）语音包生成中文语音。

## 环境准备

本项目使用 `uv` 管理 Python 虚拟环境和依赖。

```bash
uv sync
```

首次运行 Genie-TTS 时会下载基础资源和预置角色模型，文件较大，请保持网络可用。本示例会在导入 `genie_tts` 前检查 `models/GenieData`，缺失时自动从 HuggingFace 下载基础资源，不需要手动确认。Genie-TTS 官方说明中提到首次运行需要下载约 391MB 基础资源，预置角色模型会在首次加载时自动获取并保存到 `models/CharacterModels/`。

如果已经手动下载了 GenieData，可以放到 `models/GenieData`，也可以通过环境变量指定资源目录：

```powershell
$env:GENIE_DATA_DIR = "D:\path\to\GenieData"
uv run python -m text_speaker.main
```

也可以直接使用脚本参数指定，脚本会在导入 `genie_tts` 之前设置环境变量。该参数只覆盖 Genie-TTS 基础资源目录，预置语音包模型仍会保存到 `models/CharacterModels/`：

```bash
uv run python -m text_speaker.main --genie-data-dir D:\path\to\GenieData
```

## 生成中文语音

不带任何命令行参数运行时会打开 GUI 界面：

```bash
uv run python -m text_speaker.main
```

GUI 支持在多行文本框输入文本，点击“生成”或按回车开始生成；生成过程中按钮会变为“取消”，点击按钮或按回车会取消当前生成。生成完成后，如果文本框内容和最近一次生成内容一致，按回车会播放已生成语音；如果文本发生变化，则会重新生成。每次成功生成的语音都会保存在 `outputs/history/`，历史记录索引保存在 `outputs/history/history.json`。

历史记录列表支持播放和删除单条记录，历史记录面板支持清理当前列表。删除和清理只会在索引中标记记录，不会删除已生成的音频文件，也不会移除历史索引中的原始数据。

同一 GUI 进程内，同一个语音包只会加载一次，连续生成不会重复加载模型。

## 命令行生成

默认文本会生成到 `outputs/feibi_zh.wav`：

```bash
uv run python -m text_speaker.main --cli
```

自定义文本和输出路径：

```bash
uv run python -m text_speaker.main --text "你好，我是菲比。欢迎使用 Genie 生成中文语音。" --output outputs/demo.wav
```

生成后直接播放：

```bash
uv run python -m text_speaker.main --play
```

如果当前安装的 `genie-tts` 版本公开了 `download_roberta_data()`，可以额外下载 RoBERTa 文本特征资源：

```bash
uv run python -m text_speaker.main --download-roberta
```

## 参考

- Genie-TTS 支持 GPT-SoVITS V2 和 V2ProPlus，支持日语、英语、中文、韩语，Python 版本要求为 3.9 及以上。
- Genie-TTS 预置角色包含 `mika`、`thirtyseven`、`feibi`，其中 `feibi` 为中文角色。
