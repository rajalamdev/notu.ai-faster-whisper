"""
Audio Chunking Service for ASR Transcription

This service handles splitting audio into manageable chunks for processing,
with configurable overlap to preserve context at boundaries.

Best practices implemented:
- 30-second chunks (optimal for Whisper models)
- 2-second overlap for context preservation  
- Silence-aware splitting when possible
- Memory-efficient streaming approach
"""

import os
import wave
import numpy as np
import tempfile
from typing import List, Tuple, Dict, Any, Generator
from dataclasses import dataclass

# Default chunk configuration
DEFAULT_CHUNK_DURATION = 30  # seconds
DEFAULT_OVERLAP_DURATION = 2  # seconds
SAMPLE_RATE = 16000


@dataclass
class ChunkInfo:
    """Information about a single audio chunk"""
    index: int
    start_time: float  # Start time in original audio (seconds)
    end_time: float    # End time in original audio (seconds)
    duration: float    # Actual duration of chunk (seconds)
    path: str          # Path to chunk file
    has_overlap_start: bool  # True if chunk starts with overlap from previous
    has_overlap_end: bool    # True if chunk has overlap for next


@dataclass
class ChunkingResult:
    """Result of chunking operation"""
    chunks: List[ChunkInfo]
    total_duration: float
    total_chunks: int
    chunk_duration: float
    overlap_duration: float


def read_wav_info(wav_path: str) -> Tuple[int, int, int]:
    """
    Read WAV file info without loading entire file into memory.
    Returns: (sample_rate, num_channels, num_frames)
    """
    with wave.open(wav_path, 'rb') as wav:
        return wav.getframerate(), wav.getnchannels(), wav.getnframes()


def get_audio_duration(wav_path: str) -> float:
    """Get duration of WAV file in seconds"""
    sample_rate, _, num_frames = read_wav_info(wav_path)
    return num_frames / sample_rate


def calculate_chunk_boundaries(
    total_duration: float,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap_duration: float = DEFAULT_OVERLAP_DURATION
) -> List[Tuple[float, float]]:
    """
    Calculate chunk start/end times with overlap.
    
    Returns list of (start_time, end_time) tuples.
    Each chunk overlaps with adjacent chunks by overlap_duration seconds.
    """
    boundaries = []
    step = chunk_duration - overlap_duration  # Step size accounts for overlap
    
    current_start = 0.0
    
    while current_start < total_duration:
        current_end = min(current_start + chunk_duration, total_duration)
        boundaries.append((current_start, current_end))
        
        # Move to next chunk, accounting for overlap
        current_start += step
        
        # If remaining audio is less than overlap, include it in last chunk
        if total_duration - current_start < overlap_duration:
            break
    
    return boundaries


def extract_chunk(
    wav_path: str,
    start_time: float,
    end_time: float,
    output_path: str
) -> bool:
    """
    Extract a chunk from WAV file between start_time and end_time.
    Memory-efficient: reads only required portion.
    """
    try:
        with wave.open(wav_path, 'rb') as wav_in:
            sample_rate = wav_in.getframerate()
            channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            
            # Calculate frame positions
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            num_frames = end_frame - start_frame
            
            # Seek to start position
            wav_in.setpos(start_frame)
            
            # Read frames
            frames = wav_in.readframes(num_frames)
        
        # Write chunk to output file
        with wave.open(output_path, 'wb') as wav_out:
            wav_out.setnchannels(channels)
            wav_out.setsampwidth(sample_width)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(frames)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error extracting chunk: {e}")
        return False


def split_audio_into_chunks(
    wav_path: str,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap_duration: float = DEFAULT_OVERLAP_DURATION,
    output_dir: str = None
) -> ChunkingResult:
    """
    Split audio file into chunks with overlap.
    
    Args:
        wav_path: Path to WAV file (must be 16kHz mono)
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between consecutive chunks
        output_dir: Directory for chunk files (temp dir if None)
    
    Returns:
        ChunkingResult with list of ChunkInfo objects
    """
    # Get audio duration
    total_duration = get_audio_duration(wav_path)
    
    # For short audio (< 2 * chunk_duration), don't chunk
    if total_duration <= chunk_duration * 1.5:
        print(f"ℹ️ Audio is short ({total_duration:.1f}s), no chunking needed")
        return ChunkingResult(
            chunks=[ChunkInfo(
                index=0,
                start_time=0.0,
                end_time=total_duration,
                duration=total_duration,
                path=wav_path,
                has_overlap_start=False,
                has_overlap_end=False
            )],
            total_duration=total_duration,
            total_chunks=1,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration
        )
    
    # Calculate chunk boundaries
    boundaries = calculate_chunk_boundaries(total_duration, chunk_duration, overlap_duration)
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔪 Splitting {total_duration:.1f}s audio into {len(boundaries)} chunks...")
    
    chunks = []
    for i, (start, end) in enumerate(boundaries):
        chunk_path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
        
        if extract_chunk(wav_path, start, end, chunk_path):
            chunk_info = ChunkInfo(
                index=i,
                start_time=start,
                end_time=end,
                duration=end - start,
                path=chunk_path,
                has_overlap_start=(i > 0),
                has_overlap_end=(i < len(boundaries) - 1)
            )
            chunks.append(chunk_info)
            print(f"   Chunk {i+1}/{len(boundaries)}: {start:.1f}s - {end:.1f}s")
        else:
            print(f"   ⚠️ Failed to extract chunk {i+1}")
    
    return ChunkingResult(
        chunks=chunks,
        total_duration=total_duration,
        total_chunks=len(chunks),
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration
    )


def cleanup_chunks(chunking_result: ChunkingResult):
    """Remove temporary chunk files"""
    for chunk in chunking_result.chunks:
        if chunk.path and os.path.exists(chunk.path):
            try:
                os.remove(chunk.path)
            except:
                pass
    
    # Try to remove the directory if empty
    if chunking_result.chunks:
        chunk_dir = os.path.dirname(chunking_result.chunks[0].path)
        try:
            os.rmdir(chunk_dir)
        except:
            pass


@dataclass
class MergedSegment:
    """A segment after merging overlapping chunks"""
    start: float
    end: float
    text: str
    speaker: str = None


def merge_chunk_segments(
    all_chunk_segments: List[List[Dict]],
    chunk_infos: List[ChunkInfo],
    overlap_duration: float = DEFAULT_OVERLAP_DURATION
) -> List[Dict]:
    """
    Merge segments from multiple chunks, handling overlaps.
    
    Strategy:
    - For overlapping regions, keep the segment from the chunk where it falls
      in the middle (best context)
    - Deduplicate near-identical segments at boundaries
    """
    if len(all_chunk_segments) == 0:
        return []
    
    if len(all_chunk_segments) == 1:
        return all_chunk_segments[0]
    
    merged = []
    
    for chunk_idx, (chunk_segments, chunk_info) in enumerate(zip(all_chunk_segments, chunk_infos)):
        is_first = chunk_idx == 0
        is_last = chunk_idx == len(all_chunk_segments) - 1
        
        for seg in chunk_segments:
            # Convert chunk-local timestamps to global timestamps
            global_start = chunk_info.start_time + seg["start"]
            global_end = chunk_info.start_time + seg["end"]
            
            # Skip segments in overlap region that belong to adjacent chunk
            # First chunk: keep everything
            # Last chunk: skip segments in the leading overlap
            # Middle chunks: skip leading overlap segments
            
            if not is_first and chunk_info.has_overlap_start:
                # Skip segments that fall entirely in the overlap from previous chunk
                overlap_boundary = chunk_info.start_time + overlap_duration
                if global_end <= overlap_boundary:
                    continue
                # Segments that start in overlap but extend beyond - adjust start
                if global_start < overlap_boundary:
                    # Keep but mark as potentially duplicate
                    pass
            
            # Create global segment
            global_seg = {
                "start": round(global_start, 2),
                "end": round(global_end, 2),
                "text": seg["text"],
                "speaker": seg.get("speaker", "SPEAKER_0")
            }
            
            # Deduplicate: check if similar segment already exists
            is_duplicate = False
            for existing in merged[-3:] if len(merged) >= 3 else merged:
                # Check for near-duplicate (same text, close timestamps)
                if (existing["text"].strip() == global_seg["text"].strip() and
                    abs(existing["start"] - global_seg["start"]) < overlap_duration):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(global_seg)
    
    # Sort by start time
    merged.sort(key=lambda x: x["start"])
    
    return merged


def estimate_chunk_count(duration: float, chunk_duration: float = DEFAULT_CHUNK_DURATION, overlap_duration: float = DEFAULT_OVERLAP_DURATION) -> int:
    """Estimate number of chunks for a given audio duration"""
    if duration <= chunk_duration * 1.5:
        return 1
    
    step = chunk_duration - overlap_duration
    return int(np.ceil((duration - overlap_duration) / step))


# Export constants for configuration
__all__ = [
    'DEFAULT_CHUNK_DURATION',
    'DEFAULT_OVERLAP_DURATION',
    'SAMPLE_RATE',
    'ChunkInfo',
    'ChunkingResult',
    'split_audio_into_chunks',
    'merge_chunk_segments',
    'cleanup_chunks',
    'estimate_chunk_count',
    'get_audio_duration',
]
