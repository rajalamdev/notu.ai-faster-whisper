from flask import Blueprint, request, jsonify
import tempfile
import os
import time
import gc
from services.transcription_service import TranscriptionService, WHISPER_MODEL, DEVICE
from services.audio_service import convert_to_wav
from services.diarization_service import perform_simple_diarization
from services.llm_service import generate_meeting_notes

transcription_bp = Blueprint('transcription', __name__)

@transcription_bp.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe audio with speaker diarization
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
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    wav_path = None
    try:
        print(f"🎙️ Transcribing audio: {file.filename}")
        
        # Transcribe with Faster-Whisper
        segments, info = TranscriptionService.transcribe_audio(input_path, language)
        
        print(f"   Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"   Found {len(segments)} segments")
        
        # Convert to WAV for diarization if needed
        ext = input_path.split(".")[-1].lower()
        if ext != "wav":
            wav_path = input_path + ".wav"
            print(f"🔄 Converting {ext.upper()} to WAV...")
            if convert_to_wav(input_path, wav_path):
                print("✅ Conversion successful")
            else:
                print("⚠️ Conversion failed, skipping diarization")
                wav_path = None
        else:
            wav_path = input_path
        
        # Perform speaker diarization
        if wav_path and os.path.exists(wav_path):
            print("🔊 Performing speaker diarization...")
            segments, detected_speakers, silhouette_scores = perform_simple_diarization(
                wav_path, segments, num_speakers=num_speakers
            )
        else:
            # No diarization possible
            detected_speakers = 1
            silhouette_scores = {}
            for seg in segments:
                seg.speaker = "SPEAKER_0"
        
        # Build result
        segments_data = []
        for seg in segments:
            segments_data.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": getattr(seg, "speaker", "SPEAKER_0")
            })
        
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
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass
        
        print(f"✅ Transcription complete: {len(segments_data)} segments, {len(speakers)} speakers")
        
        return jsonify({
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
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
                "duration": round(info.duration, 2),
                "total_speakers": detected_speakers,
                "diarization_mode": "light-heuristic-v2",
                "model": WHISPER_MODEL,
                "device": DEVICE
            }
        })
        
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
