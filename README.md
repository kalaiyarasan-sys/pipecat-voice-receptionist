# Pipecat Voice Receptionist

A real-time voice AI receptionist for a clinic, built with [Pipecat](https://github.com/pipecat-ai/pipecat). Callers speak English, Tamil or Tanglish to book, check and cancel appointments.

```
Caller mic ─▶ Silero VAD ─▶ Whisper STT ─▶ Ollama LLM + tools ─▶ Piper TTS ─▶ Caller speaker
               (turn end)     (local,        (local, function      (local)
                               faster-whisper) calling)
```

## Features

- **100% local and free**: faster-whisper STT, Ollama LLM, Piper TTS. No API keys; works offline after a one-time model download

- **STT → LLM → TTS pipeline** with Pipecat (LLM and TTS stream; Whisper transcribes each VAD segment)
- **Barge-in**: the caller can interrupt the bot mid-sentence
- **Voice activity detection** with a tuned end-of-turn pause (`stop_secs=0.4`)
- **Tool calling**: `list_doctors`, `check_availability`, `book_appointment`, `cancel_appointment`, `transfer_to_staff`
- **Voice-first prompting**: short replies, spoken-style times, max three options, read-back confirmation before booking
- **Code-mixed speech**: replies in the caller's language (English / Tamil / Tanglish)
- **Guardrails**: no medical advice, emergency redirect, human hand-off
- **Latency metrics**: per-service TTFB logged via Pipecat metrics
- **Channel-independent business logic** (`receptionist/clinic.py`), reusable for WhatsApp or web chat, with unit tests

## Project layout

```
bot.py                   Pipecat pipeline, transports, tool registration
receptionist/clinic.py   Scheduling logic (slots, booking, cancellation, validation)
receptionist/prompts.py  System prompt for voice conversation
tests/                   Unit tests for scheduling logic
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # local model settings, no keys needed
ollama serve &            # if not already running
ollama pull qwen3:4b      # any model with "tools" capability; set LLM_MODEL in .env
python bot.py             # open http://localhost:7860 and talk to the bot
pytest                    # run tests
```

`python bot.py -t twilio` runs the same bot over a Twilio phone number (see Pipecat's Twilio docs for webhook setup).

## Local model notes

- **First run** downloads the Whisper model (Hugging Face cache) and the Piper voice (`models/piper/`). After that no internet is needed.
- **CPU latency**: `WHISPER_MODEL=base` and a 3-4B LLM keep replies around a few seconds on a laptop CPU. Use a GPU (`WHISPER_DEVICE=cuda`) for faster turns.
- **Tamil**: Whisper understands Tamil, but Piper has no Tamil voice, so the bot speaks Tamil in Roman letters (Tanglish) with an English voice.

## Roadmap

- Local Tamil TTS voice (e.g. MMS-TTS / Indic Parler-TTS) as a custom Pipecat TTS service
- PostgreSQL for bookings, Redis for live call state
- WhatsApp Business API channel sharing `clinic.py`
- Simulated-call evaluation: task success and latency per scenario
