"""Sunrise Clinic voice receptionist built with Pipecat.

Pipeline:  mic -> Silero VAD -> Whisper STT (local) -> Ollama LLM with tools
           -> Piper TTS (local) -> speaker

Everything runs on this machine: no API keys, no paid services. Models are
downloaded once (Whisper from Hugging Face, Piper voice, `ollama pull`) and
then work offline.

Run locally (opens a WebRTC test page in your browser):
    python bot.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from receptionist.clinic import BookingError, Clinic
from receptionist.prompts import GREETING_INSTRUCTION, build_system_prompt

load_dotenv(override=True)

clinic = Clinic()

# ----- tool schemas (what the LLM sees) ------------------------------------

TOOLS = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="list_doctors",
            description="List the clinic's doctors, specialties and working days.",
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="check_availability",
            description="Get up to three free appointment times for a doctor on a date.",
            properties={
                "doctor": {"type": "string", "description": "Doctor name or specialty"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD"},
                "after": {
                    "type": "string",
                    "description": "Optional earliest time, e.g. 2 PM for afternoon slots",
                },
            },
            required=["doctor", "date"],
        ),
        FunctionSchema(
            name="book_appointment",
            description="Book a slot after the caller confirmed all details.",
            properties={
                "patient_name": {"type": "string"},
                "phone": {"type": "string", "description": "10-digit mobile number"},
                "doctor": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "e.g. 10:30 or 3:00 PM"},
            },
            required=["patient_name", "phone", "doctor", "date", "time"],
        ),
        FunctionSchema(
            name="cancel_appointment",
            description="Cancel a booking using its ID and the phone number used to book.",
            properties={
                "booking_id": {"type": "string"},
                "phone": {"type": "string"},
            },
            required=["booking_id", "phone"],
        ),
        FunctionSchema(
            name="transfer_to_staff",
            description="Hand the call to a human receptionist.",
            properties={"reason": {"type": "string"}},
            required=["reason"],
        ),
    ]
)


# ----- tool handlers (what actually runs) ----------------------------------

def _safe(fn):
    """Return business errors to the LLM so it can explain them to the caller."""

    async def handler(params: FunctionCallParams):
        args = params.arguments or {}
        logger.info(f"tool {fn.__name__} args={args}")
        try:
            result = fn(**args)
        except BookingError as exc:
            result = {"error": str(exc)}
        except TypeError as exc:
            result = {"error": f"Missing or invalid details: {exc}"}
        await params.result_callback(result)

    return handler


def list_doctors():
    return {"doctors": clinic.list_doctors()}


def check_availability(doctor: str, date: str, after: str | None = None):
    return clinic.check_availability(doctor, date, after)


def book_appointment(patient_name: str, phone: str, doctor: str, date: str, time: str):
    return clinic.book(patient_name, phone, doctor, date, time)


def cancel_appointment(booking_id: str, phone: str):
    return clinic.cancel(booking_id, phone)


def transfer_to_staff(reason: str):
    # Demo stub. With a SIP/telephony transport this would issue a call transfer.
    logger.warning(f"Transfer requested: {reason}")
    return {"status": "transferring", "message": "Tell the caller a staff member will join shortly."}


HANDLERS = {
    "list_doctors": list_doctors,
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "transfer_to_staff": transfer_to_staff,
}


# ----- pipeline ------------------------------------------------------------

def _make_stt() -> WhisperSTTService:
    # "small" is multilingual (handles Tamil); use "base" for speed or
    # "large-v3-turbo" for accuracy if you have a GPU.
    # WHISPER_LANGUAGE empty = auto-detect per utterance (English/Tamil mix).
    language = os.getenv("WHISPER_LANGUAGE") or None
    return WhisperSTTService(
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        settings=WhisperSTTService.Settings(
            model=os.getenv("WHISPER_MODEL", "small"),
            language=language,
        ),
    )


def _make_llm() -> OLLamaLLMService:
    # Any Ollama model with the "tools" capability works (check: ollama show <model>).
    return OLLamaLLMService(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        settings=OLLamaLLMService.Settings(model=os.getenv("LLM_MODEL", "qwen3-4b-local")),
    )


def _make_tts() -> PiperTTSService:
    voices_dir = Path(os.getenv("PIPER_VOICES_DIR", "models/piper"))
    voices_dir.mkdir(parents=True, exist_ok=True)
    return PiperTTSService(
        download_dir=voices_dir,
        settings=PiperTTSService.Settings(voice=os.getenv("PIPER_VOICE", "en_US-lessac-medium")),
    )


def _system_prompt() -> str:
    prompt = build_system_prompt()
    # Qwen3 "thinks" before answering by default, which adds seconds of
    # latency on CPU. /no_think turns that off for a snappier call.
    if os.getenv("LLM_NO_THINK", "true").lower() == "true":
        prompt += "\n/no_think"
    return prompt


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = _make_stt()

    llm = _make_llm()
    for name, fn in HANDLERS.items():
        llm.register_function(name, _safe(fn))

    tts = _make_tts()

    messages = [{"role": "system", "content": _system_prompt()}]
    context = LLMContext(messages, TOOLS)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # barge-in: caller can talk over the bot
            enable_metrics=True,  # logs TTFB per service for latency tuning
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Caller connected")
        context.add_message({"role": "system", "content": GREETING_INSTRUCTION})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


def _audio_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # stop_secs controls how long a pause ends the caller's turn.
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
    )


transport_params = {
    "webrtc": _audio_params,
    "twilio": _audio_params,
}


async def bot(runner_args: RunnerArguments):
    """Entry point used by the Pipecat development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
