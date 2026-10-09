# Anton

A fully local, offline voice assistant inspired by Iron Man's Jarvis — built from scratch with a custom fine-tuned LLM as the action router, real-time speech pipeline, speaker verification, and an audio-reactive HUD. Everything runs on-device, no cloud APIs for the core loop.

![status](https://img.shields.io/badge/status-active-success)
![platform](https://img.shields.io/badge/platform-Linux-blue)
![python](https://img.shields.io/badge/python-3.10%2B-yellow)

## Overview

Anton listens for your voice, transcribes it, verifies it's actually you speaking, decides what to do using a fine-tuned local LLM, executes the action on your machine, and speaks the result back — all without sending anything to the cloud. It can open apps and websites, control system settings, manage files, send emails, search the web, translate text, take notes, automate git commits, read your screen, and more.

The project's core engineering challenge was building a small, fast, locally-runnable LLM that reliably converts natural speech into structured JSON actions — fast enough to feel responsive on consumer hardware, and accurate enough to trust with real system control.

## Features

- **Voice-activated control** — speak naturally, no fixed command syntax required
- **Speaker verification** — only responds to a verified voiceprint, rejects other speakers
- **Structured action routing** — a fine-tuned LLM maps free-form speech to one of 30 defined actions with typed parameters
- **System control** — volume, brightness, Wi-Fi, app launching, screen lock
- **Productivity actions** — send email, git commit & push, open VS Code on a project, take notes, set reminders
- **Information actions** — battery status, time, web search with summarized spoken answers, screen Q&A (vision model reads your screen and answers questions about it)
- **File management** — search, open, and delete files across common directories by voice
- **Translation** — on-the-fly translation between languages via local LLM
- **Conversation memory** — maintains rolling chat history across turns, persisted to disk
- **Audio-reactive HUD** — a circular pygame interface that pulses in sync with listening/speaking states

## Architecture

```
Microphone
    │
    ▼
Speaker Verification (Resemblyzer)
    │
    ▼
Speech-to-Text (Whisper, local)
    │
    ▼
Action Router (fine-tuned Qwen3-1.7B, via Ollama)
    │  → outputs structured JSON: {action, params, speak}
    ▼
Action Executor (30 Python functions: system calls, APIs, file I/O)
    │
    ▼
Text-to-Speech (Kokoro)
    │
    ▼
Speaker (with live HUD reacting to amplitude)
```

Every voice turn flows through this pipeline in real time, entirely on local hardware.

## Tech Stack

| Component | Tool |
|---|---|
| Speech-to-text | OpenAI Whisper (local inference, CPU) |
| Text-to-speech | Kokoro ONNX |
| LLM inference / serving | Ollama |
| Action router model | Fine-tuned Qwen3-1.7B (GGUF, Q4_K_M quantization) |
| Fine-tuning framework | Unsloth + LoRA |
| Speaker verification | Resemblyzer |
| Vision (screen Q&A) | minicpm-v |
| HUD | Pygame |
| Web search | DDGS (DuckDuckGo) |
| System automation | xdotool, pactl, brightnessctl, nmcli |

**Hardware used for development:** ASUS ROG Strix, RTX 3070 Ti (8GB VRAM), Ubuntu.

## Fine-Tuning the Action Router

The core intelligence of Anton isn't the speech pipeline — it's getting a small, fast model to reliably convert "hey, can you check what time it is" into `{"action": "tell_time", "params": {}, "speak": "..."}` without hallucinating actions, misparsing parameters, or wrapping output in extra commentary.

**Base model:** Qwen3-1.7B, chosen for its balance of reasoning quality and inference speed on 8GB VRAM.

**Dataset:** ~1,700 hand-built and synthetically generated examples, covering 30 distinct actions (`open_youtube`, `send_email`, `translate`, `clipboard`, `git_commit`, `screenshot`, `web_search`, `remind_me`, and more). Each example pairs a natural-language instruction with the exact JSON the model should output, including realistic phrasing variation, multi-parameter commands, and chit-chat examples that should route to `chat` rather than a system action.

**Training:** LoRA fine-tuning via Unsloth for memory-efficient training on consumer hardware, producing adapter weights merged back into the base model.

**Key engineering problems solved during fine-tuning:**
- **Thinking-mode interference** — Qwen3's reasoning/thinking tokens were leaking into the structured output and breaking JSON parsing. Resolved with `/no_think` prompting and adjustments to the training chat template so the model learned to skip the reasoning step entirely for this task.
- **GGUF conversion** — converted the merged fine-tuned weights to GGUF locally via `llama.cpp` for quantized, efficient local inference.
- **Ollama Modelfile templating** — needed careful prompt template alignment in the Modelfile to match what the model was actually trained on, otherwise output format degraded.
- **Environment isolation** — kept separate virtual environments for the Unsloth/training stack and the Ollama serving stack to avoid dependency conflicts.

**Output format:** the model is trained to always return a single JSON object:
```json
{
  "action": "send_email",
  "params": {"to": "...", "subject": "...", "body": "..."},
  "speak": "Email sent, Sir."
}
```
This gets parsed and dispatched to the corresponding Python function in `execute_actions()`.

**Quantization:** the final model is shipped as `Q4_K_M` GGUF for a fast, low-VRAM footprint suitable for real-time use alongside Whisper and Kokoro running concurrently.

## Installation

> Requires Ubuntu/Linux, Python 3.10+, Ollama installed locally, and a CUDA-capable GPU recommended for smoother inference.

```bash
git clone https://github.com/Abdelali516/anton.git
cd anton
pip install -r requirements.txt --break-system-packages
```

Pull/load the fine-tuned router model into Ollama:
```bash
ollama create jarvis -f Modelfile
```

Download the Kokoro ONNX model and voices file, and Whisper weights (downloaded automatically on first run).

Set up your voiceprint for speaker verification by recording a few short samples and generating embeddings with Resemblyzer (see `enroll_voice.py` — *if you add this script, link it here*).

## Configuration

- Update `SEARCH_DIRS` in the main script to match your filesystem layout.
- Add your contacts to `emails_contact.json` for name-based email routing.
- Set your Gmail address and an [app password](https://support.google.com/accounts/answer/185833) in `Email.txt` / `Email_password` (consider moving these to environment variables before publishing — see Security note below).

## Usage

```bash
python main.py
```

Anton will greet you on startup, then continuously listen for speech, verify it's you, and respond to commands.

## Project Structure

```
.
├── main.py            # Core assistant loop, action dispatch
├── anton_hud.py        # Pygame audio-reactive HUD
├── jarvis_conversation.json  # Persisted rolling chat history
├── jarvis_notes.txt    # Notes saved via take_note action
├── emails_contact.json # Name → email address lookup
└── requirements.txt
```

## Roadmap

- [ ] Wake word activation (no manual trigger)
- [ ] Expand action set / multi-step task chaining
- [ ] Package voice enrollment into a setup script
- [ ] Move credentials to `.env` / secrets manager
- [ ] HUD: idle animation polish, resizable window

## Security Note

This is a personal project and currently stores email credentials in plaintext local files for simplicity. If you fork or deploy this, move secrets to environment variables or a `.env` file excluded from version control before sharing your repo publicly.

## Acknowledgments

Built independently as a self-directed learning project exploring local LLM fine-tuning, real-time speech pipelines, and desktop automation.

## License

*[Add your license here, e.g. MIT]*
