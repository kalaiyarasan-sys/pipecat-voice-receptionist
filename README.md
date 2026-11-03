# Pipecat Voice Receptionist

A real-time voice AI receptionist for a clinic, built with [Pipecat](https://github.com/pipecat-ai/pipecat). Callers speak English, Tamil or Tanglish to book, check and cancel appointments.

```
Caller mic ─▶ Silero VAD ─▶ Deepgram STT ─▶ LLM + tools ─▶ Cartesia TTS ─▶ Caller speaker
               (turn end)     (streaming)    (function     (streaming)
                                              calling)
```

## Features

- **Streaming STT → LLM → TTS pipeline** with Pipecat, every stage streamed for low latency
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
cp .env.example .env      # add Deepgram, OpenAI and Cartesia keys
python bot.py             # open http://localhost:7860 and talk to the bot
pytest                    # run tests
```

`python bot.py -t twilio` runs the same bot over a Twilio phone number (see Pipecat's Twilio docs for webhook setup).

## Roadmap

- Sarvam AI STT/TTS for better Tamil recognition and voice
- PostgreSQL for bookings, Redis for live call state
- WhatsApp Business API channel sharing `clinic.py`
- Simulated-call evaluation: task success and latency per scenario
