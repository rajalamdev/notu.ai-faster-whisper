from flask import Blueprint, request, jsonify, Response
import tempfile
import os
import time
import gc
import json
import sys
from services.transcription_service import TranscriptionService, WHISPER_MODEL, DEVICE
from services.audio_service import convert_to_wav, get_audio_duration_from_file
from services.diarization_service import perform_simple_diarization
from services.llm_service import generate_meeting_notes
from services.chunk_service import estimate_chunk_count, DEFAULT_CHUNK_DURATION

transcription_bp = Blueprint('transcription', __name__)

def log(msg):
    """Print with flush for real-time output"""
    print(msg)
    sys.stdout.flush()

@transcription_bp.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe audio with speaker diarization.
    Automatically uses chunking for long audio files (> 60 seconds by default).
    """
    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    # Get parameters
    num_speakers_param = request.form.get("num_speakers", None)
    num_speakers = int(num_speakers_param) if num_speakers_param else None
    language = request.form.get("language", None)
    # Allow explicit control over chunking
    force_chunking = request.form.get("force_chunking", "").lower() == "true"
    disable_chunking = request.form.get("disable_chunking", "").lower() == "true"
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    wav_path = None
    try:
        print(f"🎙️ Transcribing audio: {file.filename}")
        
        # Get audio duration to decide on chunking
        duration = get_audio_duration_from_file(input_path)
        use_chunking = not disable_chunking and (force_chunking or TranscriptionService.should_use_chunking(duration))
        
        print(f"   Duration: {duration:.1f}s, Chunking: {'enabled' if use_chunking else 'disabled'}")
        
        # Convert to WAV first if needed (required for chunking)
        ext = input_path.split(".")[-1].lower()
        if ext != "wav":
            wav_path = input_path + ".wav"
            print(f"🔄 Converting {ext.upper()} to WAV...")
            if convert_to_wav(input_path, wav_path):
                print("✅ Conversion successful")
            else:
                print("⚠️ Conversion failed, proceeding with original file")
                wav_path = None
        else:
            wav_path = input_path
        
        # Choose transcription method based on duration
        chunking_metadata = None
        if use_chunking and wav_path:
            print(f"🔪 Using chunked transcription for {duration:.1f}s audio")
            segments_data, info, chunking_metadata = TranscriptionService.transcribe_audio_chunked(
                wav_path, language
            )
            # segments_data is already list of dicts
            segments = segments_data
        else:
            # Original method
            segments_list, info = TranscriptionService.transcribe_audio(
                wav_path or input_path, language
            )
            # Convert to segments_data format
            segments = []
            for seg in segments_list:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
        
        print(f"   Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"   Found {len(segments)} segments")
        
        # Perform speaker diarization
        if wav_path and os.path.exists(wav_path):
            print("🔊 Performing speaker diarization...")
            # For chunked transcription, segments are dicts; diarization expects segment objects
            # Create wrapper objects for diarization
            class SegmentWrapper:
                def __init__(self, d):
                    self.start = d["start"]
                    self.end = d["end"]
                    self.text = d["text"]
                    self.speaker = d.get("speaker", "SPEAKER_0")
            
            seg_objects = [SegmentWrapper(s) for s in segments]
            seg_objects, detected_speakers, silhouette_scores = perform_simple_diarization(
                wav_path, seg_objects, num_speakers=num_speakers
            )
            
            # Convert back to dicts with speaker info
            segments = []
            for seg in seg_objects:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "speaker": getattr(seg, "speaker", "SPEAKER_0")
                })
        else:
            # No diarization possible
            detected_speakers = 1
            silhouette_scores = {}
            for seg in segments:
                seg["speaker"] = "SPEAKER_0"
        
        segments_data = segments
        
        # Generate full transcript
        transcript_text = " ".join([s["text"] for s in segments_data])
        
        # Generate structured meeting notes
        print("📝 Generating structured meeting notes...")
        meeting_notes = generate_meeting_notes(transcript_text)
        
        # Get unique speakers
        speakers = list(set([s["speaker"] for s in segments_data]))
        speakers.sort()
        
        # Force garbage collection
        gc.collect()
        
        # Cleanup temp files
        for path in [input_path, wav_path]:
            if path and os.path.exists(path) and path != input_path:
                try:
                    os.remove(path)
                except PermissionError:
                    pass
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except PermissionError:
                pass
        
        print(f"✅ Transcription complete: {len(segments_data)} segments, {len(speakers)} speakers")
        
        response_data = {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(duration, 2),  # Use original audio duration, not info.duration (may be chunk duration)
            "segments": segments_data,
            "speakers": speakers,
            "num_speakers_detected": detected_speakers,
            "silhouette_scores": silhouette_scores,
            "transcript": transcript_text,
            "suggestedTitle": meeting_notes.get("suggestedTitle", ""),
            "suggestedDescription": meeting_notes.get("suggestedDescription", ""),
            "summary": meeting_notes.get("summary", ""),
            "highlights": meeting_notes.get("highlights", {}),
            "tags": meeting_notes.get("tags", []),
            "action_items": meeting_notes.get("actionItems", []),
            "conclusion": meeting_notes.get("conclusion", ""),
            "model": WHISPER_MODEL,
            "device": DEVICE,
            "metadata": {
                "duration": round(duration, 2),  # Use original audio duration
                "total_speakers": detected_speakers,
                "diarization_mode": "light-heuristic-v2",
                "model": WHISPER_MODEL,
                "device": DEVICE,
                "chunking": chunking_metadata or {"chunking_used": False}
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        # Cleanup on error
        try:
            os.remove(input_path)
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass
        
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({"error": str(e)}), 500


@transcription_bp.route("/transcribe/stream", methods=["POST"])
def transcribe_stream():
    """
    Transcribe audio with Server-Sent Events (SSE) for real-time progress.
    Reports chunk-by-chunk progress for long audio files.
    
    Returns SSE stream with events:
    - progress: {chunk, total_chunks, stage, message}
    - transcript_chunk: {chunk_index, segments} 
    - complete: {full response data}
    - error: {error message}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    num_speakers_param = request.form.get("num_speakers", None)
    num_speakers = int(num_speakers_param) if num_speakers_param else None
    language = request.form.get("language", None)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    def generate():
        wav_path = None
        try:
            # Stage: starting (0-9)
            log(f"📥 [START] File received, starting processing...")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'starting', 'message': 'File diterima, memulai proses...', 'progress': 5})}\n\n"
            
            duration = get_audio_duration_from_file(input_path)
            estimated_chunks = estimate_chunk_count(duration)
            
            # Stage: downloading/processing (10-19)
            log(f"📊 [ANALYZE] Duration: {duration:.1f}s, estimated chunks: {estimated_chunks}")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'downloading', 'message': f'Durasi: {duration:.0f} detik', 'progress': 12})}\n\n"
            
            # Convert to WAV
            ext = input_path.split(".")[-1].lower()
            if ext != "wav":
                wav_path = input_path + ".wav"
                log(f"🔄 [CONVERT] Converting to WAV format...")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'downloading', 'message': 'Mengkonversi format audio...', 'progress': 15})}\n\n"
                convert_to_wav(input_path, wav_path)
                log(f"✓ [CONVERT] WAV conversion complete")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'downloading', 'message': 'Konversi selesai', 'progress': 19})}\n\n"
            else:
                wav_path = input_path
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'downloading', 'message': 'File siap diproses', 'progress': 19})}\n\n"
            
            # Progress callback for chunk transcription
            def on_chunk_progress(chunk_idx, total_chunks, status):
                progress = 20 + int((chunk_idx / total_chunks) * 50)  # 20-70%
                return f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'chunk': chunk_idx, 'total_chunks': total_chunks, 'message': status, 'progress': progress})}\n\n"
            
            # Stage: transcribing (20-69)
            segments_data = []
            info = None
            chunking_metadata = None
            
            if TranscriptionService.should_use_chunking(duration):
                # Chunked transcription for long audio
                log(f"📝 [TRANSCRIBE] Starting chunked transcription for {duration:.1f}s audio...")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'message': 'Memulai transkripsi chunked...', 'progress': 20})}\n\n"
                
                from services.chunk_service import split_audio_into_chunks, merge_chunk_segments, cleanup_chunks
                
                chunking_result = split_audio_into_chunks(wav_path)
                all_chunk_segments = []
                total = chunking_result.total_chunks
                
                for chunk in chunking_result.chunks:
                    chunk_num = chunk.index + 1
                    # Progress 22-62 for chunks (leaving room for merge)
                    progress = 22 + int((chunk_num / total) * 40)
                    
                    log(f"   🎯 [TRANSCRIBE] Processing chunk {chunk_num}/{total}...")
                    yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'chunk': chunk_num, 'total_chunks': total, 'message': f'Transkripsi chunk {chunk_num}/{total}...', 'progress': progress})}\n\n"
                    
                    chunk_segments, chunk_info = TranscriptionService.transcribe_chunk(chunk.path, language)
                    
                    if info is None:
                        info = chunk_info
                    
                    all_chunk_segments.append(chunk_segments)
                    log(f"   ✓ [TRANSCRIBE] Chunk {chunk_num}/{total} done ({len(chunk_segments)} segments)")
                    
                    yield f"data: {json.dumps({'type': 'transcript_chunk', 'chunk_index': chunk.index, 'segments': chunk_segments, 'start_time': chunk.start_time})}\n\n"
                    
                    gc.collect()
                
                # Merging chunks
                log(f"📦 [TRANSCRIBE] Merging {total} chunks...")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'message': 'Menggabungkan chunk...', 'progress': 65})}\n\n"
                
                segments_data = merge_chunk_segments(all_chunk_segments, chunking_result.chunks)
                chunking_metadata = {"chunking_used": True, "total_chunks": total}
                cleanup_chunks(chunking_result)
                
                log(f"✓ [TRANSCRIBE] Merge complete: {len(segments_data)} total segments")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'message': f'Transkripsi selesai ({len(segments_data)} segmen)', 'progress': 69})}\n\n"
            else:
                # Direct transcription for short audio
                log(f"📝 [TRANSCRIBE] Starting direct transcription for {duration:.1f}s audio...")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'message': 'Memproses transkripsi...', 'progress': 25})}\n\n"
                
                segments_list, info = TranscriptionService.transcribe_audio(wav_path or input_path, language)
                segments_data = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments_list]
                chunking_metadata = {"chunking_used": False}
                
                log(f"✓ [TRANSCRIBE] Complete: {len(segments_data)} segments")
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'transcribing', 'message': f'Transkripsi selesai ({len(segments_data)} segmen)', 'progress': 69})}\n\n"
            
            # Stage: diarization (70-79)
            log(f"👥 [DIARIZATION] Starting speaker identification...")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'diarization', 'message': 'Mengidentifikasi pembicara...', 'progress': 70})}\n\n"
            
            class SegmentWrapper:
                def __init__(self, d):
                    self.start = d["start"]
                    self.end = d["end"]
                    self.text = d["text"]
                    self.speaker = d.get("speaker", "SPEAKER_0")
            
            seg_objects = [SegmentWrapper(s) for s in segments_data]
            seg_objects, detected_speakers, silhouette_scores = perform_simple_diarization(
                wav_path, seg_objects, num_speakers=num_speakers
            )
            
            segments_data = [{
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": getattr(seg, "speaker", "SPEAKER_0")
            } for seg in seg_objects]
            
            log(f"✓ [DIARIZATION] Complete: {detected_speakers} speakers detected")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'diarization', 'message': f'{detected_speakers} pembicara terdeteksi', 'progress': 79})}\n\n"
            
            # Stage: ai_analysis (80-89)
            log(f"🤖 [AI] Generating meeting notes...")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'ai_analysis', 'message': 'Membuat ringkasan meeting...', 'progress': 80})}\n\n"
            
            transcript_text = " ".join([s["text"] for s in segments_data])
            meeting_notes = generate_meeting_notes(transcript_text)
            
            speakers = sorted(list(set([s["speaker"] for s in segments_data])))
            action_items_count = len(meeting_notes.get("actionItems", []))
            summary_len = len(meeting_notes.get('summary', ''))
            
            log(f"✓ [AI] Meeting notes generated: {summary_len} chars, {action_items_count} action items")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'ai_analysis', 'message': f'Ringkasan dibuat ({action_items_count} action items)', 'progress': 89})}\n\n"
            
            # Stage: saving (90-99)
            log(f"💾 [SAVING] Finalizing results...")
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'saving', 'message': 'Menyimpan hasil...', 'progress': 90})}\n\n"
            
            # Final result - use original 'duration' variable, not info.duration (which may be chunk duration)
            result = {
                "type": "complete",
                "language": info.language,
                "duration": round(duration, 2),  # Use original audio duration
                "segments": segments_data,
                "speakers": speakers,
                "num_speakers_detected": detected_speakers,
                "transcript": transcript_text,
                "suggestedTitle": meeting_notes.get("suggestedTitle", ""),
                "suggestedDescription": meeting_notes.get("suggestedDescription", ""),
                "summary": meeting_notes.get("summary", ""),
                "highlights": meeting_notes.get("highlights", {}),
                "tags": meeting_notes.get("tags", []),
                "action_items": meeting_notes.get("actionItems", []),
                "conclusion": meeting_notes.get("conclusion", ""),
                "metadata": {
                    "model": WHISPER_MODEL,
                    "device": DEVICE,
                    "duration": round(duration, 2),  # Use original audio duration
                    "total_speakers": detected_speakers,
                    "diarization_mode": "light-heuristic-v2",
                    "chunking": chunking_metadata
                }
            }
            
            # Emit saving complete before final result
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'saving', 'message': 'Finalisasi selesai', 'progress': 99})}\n\n"
            
            log(f"✅ [COMPLETE] Transcription finished! Duration: {duration:.1f}s, Speakers: {detected_speakers}, Segments: {len(segments_data)}")
            
            # Stage: completed (100)
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'completed', 'message': f'Selesai! {len(segments_data)} segmen, {detected_speakers} pembicara', 'progress': 100})}\n\n"
            
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            import traceback
            log(f"❌ [ERROR] {str(e)}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        finally:
            # Cleanup
            for path in [input_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            gc.collect()
    
    # Return SSE response with proper headers to disable buffering
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@transcription_bp.route("/config", methods=["GET"])
def get_config():
    """Return current transcription configuration including chunking settings"""
    return jsonify({
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "chunking": TranscriptionService.get_chunking_config()
    })


@transcription_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Generate structured meeting notes from existing transcript
    """
    data = request.json
    if not data or "transcript" not in data:
        return jsonify({"error": "No transcript provided"}), 400
    
    transcript_text = data["transcript"]
    
    try:
        print("📝 Regenerating meeting notes from transcript...")
        meeting_notes = generate_meeting_notes(transcript_text)
        
        return jsonify({
            "suggestedTitle": meeting_notes.get("suggestedTitle", ""),
            "suggestedDescription": meeting_notes.get("suggestedDescription", ""),
            "summary": meeting_notes.get("summary", ""),
            "highlights": meeting_notes.get("highlights", {}),
            "tags": meeting_notes.get("tags", []),
            "action_items": meeting_notes.get("actionItems", []),
            "conclusion": meeting_notes.get("conclusion", ""),
        })
    except Exception as e:
        # Return a structured fallback response instead of HTTP 500 so callers
        # (backend) can handle LLM parsing failures gracefully.
        print(f"❌ Analysis Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "suggestedTitle": "",
            "suggestedDescription": "",
            "summary": "Ringkasan tidak tersedia (error saat memproses).",
            "highlights": {},
            "tags": [],
            "action_items": [],
            "conclusion": "",
            "__llm_diagnostics": {"error": str(e), "fallback": True}
        }), 200
