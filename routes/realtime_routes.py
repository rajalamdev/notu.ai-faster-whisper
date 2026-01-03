from flask import Blueprint, request, jsonify
import tempfile
import os
import sys
import time
from services.transcription_service import TranscriptionService
from services.audio_service import convert_to_wav_ffmpeg

realtime_bp = Blueprint('realtime', __name__)

def log(msg):
    """Print with flush for real-time output"""
    print(msg)
    sys.stdout.flush()

@realtime_bp.route("/transcribe/realtime", methods=["POST"])
def transcribe_realtime():
    """
    Transcribe audio chunk for realtime preview.
    Optimized for speed over accuracy - uses VAD and minimal processing.
    
    Expects:
    - file: Audio chunk (webm, wav, etc.)
    - session_id: Unique session identifier
    - is_final: Boolean indicating if this is the final chunk for post-processing
    
    Returns:
    - text: Transcribed text
    - is_partial: Whether this is a partial result
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    session_id = request.form.get("session_id", "unknown")
    is_final = request.form.get("is_final", "false").lower() == "true"
    language = request.form.get("language", None)
    
    # Save uploaded chunk
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    wav_path = None
    start_time = time.time()
    
    try:
        log(f"🎤 Realtime transcription - Session: {session_id}, Final: {is_final}")
        
        # Convert to WAV for Whisper using ffmpeg (more robust for webm chunks)
        wav_converted = False
        if ext.lower() != "wav":
            wav_path = input_path + ".wav"
            if convert_to_wav_ffmpeg(input_path, wav_path):
                wav_converted = True
            else:
                # If conversion fails, try with original
                log("⚠️ WAV conversion failed, trying original file")
                wav_path = input_path
        else:
            wav_path = input_path
            wav_converted = True
        
        # Check if the file is readable by Whisper
        if not wav_converted and not os.path.exists(wav_path):
            return jsonify({
                "text": "",
                "segments": [],
                "language": "id",
                "language_probability": 0,
                "is_partial": not is_final,
                "session_id": session_id,
                "processing_time": 0,
                "warning": "Could not convert audio file"
            })
        
        # Transcribe using the cached model
        # For realtime, we use simpler settings for speed
        model = TranscriptionService.get_model()
        segments_generator, info = model.transcribe(
            wav_path,
            language=language or os.getenv("LANGUAGE", "id"),
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,  # Shorter for realtime responsiveness
                "speech_pad_ms": 100,
            },
            beam_size=1 if not is_final else 5,  # Faster for preview, more accurate for final
            best_of=1 if not is_final else 5,
        )
        
        # Collect segments
        segments = []
        full_text = ""
        for seg in segments_generator:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text += seg.text.strip() + " "
        
        processing_time = time.time() - start_time
        log(f"✅ Transcribed in {processing_time:.2f}s: {full_text[:100]}...")
        
        return jsonify({
            "text": full_text.strip(),
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "is_partial": not is_final,
            "session_id": session_id,
            "processing_time": round(processing_time, 2),
        })
        
    except Exception as e:
        log(f"❌ Realtime transcription error: {str(e)}")
        return jsonify({
            "error": str(e),
            "session_id": session_id,
        }), 500
        
    finally:
        # Cleanup temp files
        try:
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            log(f"⚠️ Cleanup error: {e}")


@realtime_bp.route("/transcribe/realtime/final", methods=["POST"])
def transcribe_realtime_final():
    """
    Final transcription with diarization for realtime session.
    Called when user stops recording - processes full audio with higher quality settings.
    
    Expects:
    - file: Complete audio file
    - session_id: Session identifier
    - num_speakers: Expected number of speakers (optional)
    
    Returns:
    - Full transcription result with segments and diarization
    """
    from services.diarization_service import perform_simple_diarization
    from services.llm_service import generate_meeting_notes
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    session_id = request.form.get("session_id", "unknown")
    num_speakers_param = request.form.get("num_speakers", None)
    num_speakers = int(num_speakers_param) if num_speakers_param else None
    language = request.form.get("language", None)
    enable_ai_notes = request.form.get("enable_ai_notes", "true").lower() == "true"
    
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    wav_path = None
    start_time = time.time()
    
    try:
        log(f"🎤 Final transcription for session: {session_id}")
        
        # Convert to WAV using ffmpeg (more robust for webm)
        wav_converted = False
        if ext.lower() != "wav":
            wav_path = input_path + ".wav"
            if convert_to_wav_ffmpeg(input_path, wav_path):
                wav_converted = True
            else:
                wav_path = input_path
                log("⚠️ WAV conversion failed, using original file")
        else:
            wav_path = input_path
            wav_converted = True
        
        # Use higher quality settings for final transcription
        from services.audio_service import get_audio_duration_from_file
        duration = get_audio_duration_from_file(wav_path)
        log(f"   Audio duration: {duration:.1f}s")
        
        # Decide chunking based on duration
        use_chunking = duration > 300  # 5 minutes
        
        if use_chunking:
            segments, info, chunking_metadata = TranscriptionService.transcribe_audio_chunked(
                wav_path, language
            )
        else:
            segments_list, info = TranscriptionService.transcribe_audio(
                wav_path, language
            )
            segments = []
            for seg in segments_list:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
        
        # Perform diarization (only if we have a valid WAV file)
        if wav_converted:
            log("🔊 Performing speaker diarization...")
            diarization_result = perform_simple_diarization(
                wav_path, segments, num_speakers
            )
        else:
            log("⚠️ Skipping diarization (no valid WAV file)")
            # Fallback: assign all segments to SPEAKER_0
            diarized_segments = []
            for seg in segments:
                diarized_segments.append({
                    **seg,
                    "speaker": "SPEAKER_0"
                })
            diarization_result = {
                "segments": diarized_segments,
                "speaker_timeline": {"SPEAKER_0": sum(s["end"] - s["start"] for s in segments)},
                "num_speakers": 1,
                "method": "fallback (no wav)"
            }
        
        # Build transcript
        transcript = " ".join([s["text"] for s in diarization_result["segments"]])
        
        # Generate AI meeting notes if enabled
        ai_notes = None
        if enable_ai_notes and transcript:
            log("🤖 Generating AI meeting notes...")
            try:
                ai_notes = generate_meeting_notes(transcript)
            except Exception as e:
                log(f"⚠️ AI notes generation failed: {e}")
        
        processing_time = time.time() - start_time
        log(f"✅ Final transcription completed in {processing_time:.1f}s")
        
        return jsonify({
            "session_id": session_id,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "transcript": transcript,
            "segments": diarization_result["segments"],
            "speakers": diarization_result["speaker_timeline"],
            "num_speakers": diarization_result["num_speakers"],
            "diarization_method": diarization_result["method"],
            "duration": round(duration, 2),
            "processing_time": round(processing_time, 2),
            "ai_notes": ai_notes,
            "metadata": {
                "chunking_used": use_chunking,
            }
        })
        
    except Exception as e:
        log(f"❌ Final transcription error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "session_id": session_id,
        }), 500
        
    finally:
        try:
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            log(f"⚠️ Cleanup error: {e}")
