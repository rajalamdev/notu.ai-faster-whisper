import os
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

class TranscriptionService:
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            print(f"🔄 Loading Faster-Whisper model: {WHISPER_MODEL}")
            cls._instance = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
            print("✅ Model loaded successfully!")
        return cls._instance

    @staticmethod
    def transcribe_audio(input_path, language=None):
        model = TranscriptionService.get_model()
        segments_generator, info = model.transcribe(
            input_path,
            language=language or os.getenv("LANGUAGE", "id"),
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
            }
        )
        return list(segments_generator), info
