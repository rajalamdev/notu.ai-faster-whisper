import os
import gc
from typing import List, Dict, Any, Tuple, Generator, Callable, Optional
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from services.chunk_service import (
    split_audio_into_chunks,
    merge_chunk_segments,
    cleanup_chunks,
    ChunkingResult,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_OVERLAP_DURATION,
    estimate_chunk_count
)

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

# Chunking configuration
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", DEFAULT_CHUNK_DURATION))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_OVERLAP_DURATION))
# Audio duration threshold for chunking (in seconds) - set high to avoid overhead for short files
CHUNKING_THRESHOLD = int(os.getenv("CHUNKING_THRESHOLD", 300))  # 5 minutes


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
        """Original transcription method - processes entire file at once"""
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

    @staticmethod
    def transcribe_chunk(chunk_path: str, language: str = None) -> Tuple[List[Dict], Any]:
        """
        Transcribe a single chunk of audio.
        Returns list of segment dicts with local timestamps.
        """
        model = TranscriptionService.get_model()
        segments_generator, info = model.transcribe(
            chunk_path,
            language=language or os.getenv("LANGUAGE", "id"),
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,  # Shorter for chunks
            }
        )
        
        # Convert to list of dicts for easier merging
        segments = []
        for seg in segments_generator:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
        
        return segments, info

    @staticmethod
    def transcribe_audio_chunked(
        wav_path: str,
        language: str = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        chunk_duration: float = None,
        overlap_duration: float = None
    ) -> Tuple[List[Dict], Any, Dict]:
        """
        Transcribe audio using chunking approach for long files.
        
        Args:
            wav_path: Path to WAV file (must be 16kHz mono)
            language: Language code (e.g., 'id', 'en')
            progress_callback: Optional callback(chunk_idx, total_chunks, status)
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            (merged_segments, info, metadata)
        """
        chunk_dur = chunk_duration or CHUNK_DURATION
        overlap_dur = overlap_duration or CHUNK_OVERLAP
        
        # Split audio into chunks
        print(f"🔪 Starting chunked transcription (chunk={chunk_dur}s, overlap={overlap_dur}s)")
        chunking_result = split_audio_into_chunks(
            wav_path,
            chunk_duration=chunk_dur,
            overlap_duration=overlap_dur
        )
        
        if chunking_result.total_chunks == 1:
            # No chunking needed, use original file
            print("ℹ️ Single chunk - using direct transcription")
            segments, info = TranscriptionService.transcribe_audio(wav_path, language)
            segments_data = [{
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            } for seg in segments]
            
            return segments_data, info, {
                "chunking_used": False,
                "total_chunks": 1,
                "total_duration": chunking_result.total_duration
            }
        
        all_chunk_segments = []
        info = None
        detected_language = None
        
        try:
            print(f"📊 Processing {chunking_result.total_chunks} chunks...")
            
            for chunk in chunking_result.chunks:
                chunk_num = chunk.index + 1
                total = chunking_result.total_chunks
                
                if progress_callback:
                    progress_callback(chunk_num, total, f"Transcribing chunk {chunk_num}/{total}")
                
                print(f"   🎙️ Chunk {chunk_num}/{total}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
                
                # Transcribe this chunk
                chunk_segments, chunk_info = TranscriptionService.transcribe_chunk(chunk.path, language)
                
                # Store info from first chunk
                if info is None:
                    info = chunk_info
                    detected_language = chunk_info.language
                
                all_chunk_segments.append(chunk_segments)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Merge all chunk segments
            print("🔗 Merging chunk segments...")
            if progress_callback:
                progress_callback(chunking_result.total_chunks, chunking_result.total_chunks, "Merging segments")
            
            merged_segments = merge_chunk_segments(
                all_chunk_segments,
                chunking_result.chunks,
                overlap_duration=overlap_dur
            )
            
            print(f"✅ Chunked transcription complete: {len(merged_segments)} segments from {chunking_result.total_chunks} chunks")
            
            metadata = {
                "chunking_used": True,
                "total_chunks": chunking_result.total_chunks,
                "chunk_duration": chunk_dur,
                "overlap_duration": overlap_dur,
                "total_duration": chunking_result.total_duration,
                "segments_per_chunk": [len(segs) for segs in all_chunk_segments]
            }
            
            return merged_segments, info, metadata
            
        finally:
            # Always cleanup chunk files
            cleanup_chunks(chunking_result)
            gc.collect()

    @staticmethod
    def should_use_chunking(duration: float) -> bool:
        """Determine if chunking should be used based on duration"""
        if not ENABLE_CHUNKING:
            return False
        return duration > CHUNKING_THRESHOLD

    @staticmethod
    def get_chunking_config() -> Dict[str, Any]:
        """Return current chunking configuration"""
        return {
            "enabled": ENABLE_CHUNKING,
            "chunk_duration": CHUNK_DURATION,
            "overlap_duration": CHUNK_OVERLAP,
            "threshold_seconds": CHUNKING_THRESHOLD,
            "model": WHISPER_MODEL,
            "device": DEVICE
        }
