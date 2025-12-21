# Faster-Whisper + Simple Diarization API
# Optimized for CPU (8GB RAM) with Bahasa Indonesia support

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import tempfile
import os
import time
import requests
import gc
import numpy as np
from faster_whisper import WhisperModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import wave
import av  # PyAV - already installed with faster-whisper
import requests
import json


# Load environment variables
load_dotenv()

# ======================================================
# CONFIGURATION
# ======================================================
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
LANGUAGE = os.getenv("LANGUAGE", "id")  # Indonesian by default

print("🔄 Loading Faster-Whisper model...")
print(f"   Model: {WHISPER_MODEL}, Device: {DEVICE}, Compute: {COMPUTE_TYPE}, Language: {LANGUAGE}")

# Load Faster-Whisper model once at startup
WHISPER = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

print("✅ Model loaded successfully!")

# ======================================================
# HELPERS
# ======================================================

def convert_to_wav(input_path, output_path):
    """Convert any audio/video to WAV using PyAV"""
    container = None
    output = None
    try:
        # Open input file
        container = av.open(input_path)
        
        # Open output WAV file with mono layout
        output = av.open(output_path, 'w')
        out_stream = output.add_stream('pcm_s16le', rate=16000, layout='mono')
        
        # Resample and convert
        resampler = av.audio.resampler.AudioResampler(
            format='s16',
            layout='mono',
            rate=16000
        )
        
        for frame in container.decode(audio=0):
            frame.pts = None
            resampled_frames = resampler.resample(frame)
            # resample() returns list of frames in newer PyAV
            if isinstance(resampled_frames, list):
                for resampled_frame in resampled_frames:
                    for packet in out_stream.encode(resampled_frame):
                        output.mux(packet)
            else:
                for packet in out_stream.encode(resampled_frames):
                    output.mux(packet)
        
        # Flush
        for packet in out_stream.encode(None):
            output.mux(packet)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Conversion error: {e}")
        return False
    finally:
        # Always close handles
        if container:
            container.close()
        if output:
            output.close()


def auto_detect_speakers(features, min_speakers=2, max_speakers=6):
    """Auto-detect optimal number of speakers using Silhouette Score"""
    if len(features) < 2:
        return 1, {}
    
    # Normalize features
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Adjust range based on available data
    max_k = min(max_speakers, len(features) - 1)
    min_k = min(min_speakers, max_k)
    
    # If not enough data for clustering, return 1 speaker
    if min_k >= max_k or max_k < 2:
        return 1, {}
    
    best_score = -1
    best_k = min_k
    scores = {}
    
    # Test different values of k
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_norm)
        score = silhouette_score(features_norm, labels)
        scores[k] = round(score, 4)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k, scores


def extract_audio_features(audio_data, sample_rate=16000):
    """Extract simple audio features for speaker clustering"""
    # Calculate energy in different frequency bands
    n_samples = len(audio_data)
    if n_samples == 0:
        return np.zeros(3)
    
    # Simple features: mean, std, energy
    mean_val = np.mean(audio_data)
    std_val = np.std(audio_data)
    energy = np.sum(audio_data ** 2) / n_samples
    
    return np.array([mean_val, std_val, energy])


def perform_simple_diarization(audio_path, segments, num_speakers=None):
    """
    Perform simple speaker diarization using audio features + KMeans clustering
    
    Args:
        audio_path: Path to audio file
        segments: Whisper transcription segments
        num_speakers: Number of speakers (None = auto-detect)
    
    Returns:
        segments with speaker labels added
    """
    try:
        # Load audio file
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            audio_bytes = wf.readframes(wf.getnframes())
            
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)[:, 0]  # Use first channel
        
        # Normalize
        audio_data = audio_data / 32768.0
        
        # Extract features for each segment
        features = []
        valid_segments = []
        
        for seg in segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            
            # Extract audio chunk
            chunk = audio_data[start_sample:end_sample]
            
            # Skip very short segments
            if len(chunk) < sample_rate * 0.3:  # < 0.3 seconds
                continue
            
            # Extract features
            feat = extract_audio_features(chunk, sample_rate)
            features.append(feat)
            valid_segments.append(seg)
        
        if len(features) < 2:
            # Not enough segments for clustering
            for seg in segments:
                seg.speaker = "SPEAKER_0"
            return segments, 1, {}
        
        # Convert to numpy array and normalize
        features = np.array(features)
        features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # AUTO-DETECT or USE SPECIFIED
        if num_speakers is None or num_speakers == 0:
            n_clusters, scores = auto_detect_speakers(features)
            print(f"🔍 Auto-detected {n_clusters} speakers")
            print(f"   Silhouette scores: {scores}")
        else:
            n_clusters = min(num_speakers, len(features))
            scores = {}
        
        # Skip clustering if n_clusters is 0 or 1
        if n_clusters <= 1:
            for seg in segments:
                seg.speaker = "SPEAKER_0"
            return segments, 1, {}
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_norm)
        
        # Assign speaker labels to segments
        label_idx = 0
        for seg in segments:
            if label_idx < len(valid_segments) and seg.start == valid_segments[label_idx].start:
                seg.speaker = f"SPEAKER_{labels[label_idx]}"
                label_idx += 1
            else:
                # For skipped short segments, assign to previous speaker
                seg.speaker = f"SPEAKER_{labels[min(label_idx-1, len(labels)-1)] if label_idx > 0 else 0}"
        
        return segments, n_clusters, scores
        
    except Exception as e:
        print(f"⚠️ Diarization error: {str(e)}")
        # Fallback: assign all to single speaker
        for seg in segments:
            seg.speaker = "SPEAKER_0"
        return segments, 1, {}


def generate_meeting_notes(transcript_text):
    """Generate structured meeting notes using OpenRouter API with professional prompting"""
    if not OPENROUTER_KEY:
        return {
            "summary": "Ringkasan tidak tersedia (API key tidak dikonfigurasi).",
            "highlights": {},
            "actionItems": [],
            "conclusion": ""
        }
    
    try:
        prompt = """Kamu adalah sekretaris rapat profesional. 
Tugasmu adalah membuat notulensi rapat dalam bahasa Indonesia berdasarkan transkrip berikut.

⚠️ Aturan Penting:
1. Struktur output wajib (Strict JSON):
   - Summary → ringkasan singkat (2–3 paragraf) dalam format markdown. Gunakan **bold** untuk poin penting.
   - Highlights → catatan detail berdasarkan topik utama dalam format OBJECT (Key-Value Dictionary). JANGAN gunakan Array/List.
       • Key: Sub-judul topik (misal: "Gambaran Umum", "Budget", "Next Steps").
       • Value: Isi catatan detail dalam format markdown (gunakan bullet points `- ` dan **bold**).
       • Contoh salah: "highlights": ["catatan 1", "catatan 2"]
       • Contoh benar: "highlights": { "Topik A": "- Poin 1\n- Poin 2", "Topik B": "Paragraf penjelasan." }
   - Action Items → daftar tugas (Array of Objects).
       • Wajib: title, description (detail), priority (low/medium/high/urgent).
       • Assignee: selalu null. Labels: array string. Status: "todo".
   - Conclusion → penutup rapat (Markdown).

2. Gaya Bahasa: Formal, profesional, jelas.
3. Markdown Usage: Gunakan syntax markdown (bold, italic, lists) di dalam string value JSON agar tampilan rapi.

Format output JSON HARUS persis seperti contoh ini:
{
  "summary": "**Rapat ini** membahas tentang strategi pemasaran Q3. Disepakati bahwa...",
  "highlights": {
    "Strategi Pemasaran": "- Fokus pada **media sosial**.\n- Budget dialokasikan sebesar Rp 50jt.",
    "Timeline": "- Peluncuran: **1 Agustus**.\n- Evaluasi: **15 Agustus**."
  },
  "actionItems": [
    {
      "title": "Siapkan materi kampanye",
      "description": "Buat materi visual untuk Instagram dan TikTok sesuai guideline baru.",
      "assignee": null,
      "priority": "high",
      "dueDate": "2025-08-01",
      "status": "todo",
      "labels": ["marketing", "q3"]
    }
  ],
  "conclusion": "**Kesimpulan rapat**: Kampanye Q3 akan dimulai tepat waktu dengan fokus digital."
}

Hanya kembalikan JSON, tanpa penjelasan tambahan. PASTIKAN semua field diisi dengan konten yang relevan, dan pastikan semuanya digenerate sesuai perintah.

Transkripsi:
""" + transcript_text


        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                # "HTTP-Referer": "http://localhost:5005",
                # "X-Title": "Notu.ai Transcription"
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            # Remove markdown code blocks
            lines = content.split("\n")
            json_start = None
            json_end = None
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if json_start is None:
                        json_start = i + 1
                    else:
                        json_end = i
                        break
            if json_start and json_end:
                content = "\n".join(lines[json_start:json_end])
        
        # Parse JSON
        notes_data = json.loads(content)
        
        # Ensure structure
        result = {
            "summary": notes_data.get("summary", ""),
            "highlights": notes_data.get("highlights", {}),
            "actionItems": notes_data.get("actionItems", []),
            "conclusion": notes_data.get("conclusion", "")
        }
        
        # Debug: print result to see what we got
        print(f"📋 Generated meeting notes:")
        print(f"   Summary length: {len(result['summary'])}")
        print(f"   Highlights keys: {list(result['highlights'].keys()) if isinstance(result['highlights'], dict) else 'N/A'}")
        print(f"   Action items count: {len(result['actionItems'])}")
        print(f"   Conclusion length: {len(result['conclusion'])}")
        
        return result
    except Exception as e:
        print(f"⚠️ Meeting notes generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "summary": "Ringkasan tidak tersedia.",
            "highlights": {},
            "actionItems": [],
            "conclusion": ""
        }


# ======================================================
# FLASK API
# ======================================================

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "faster-whisper",
        "model": WHISPER_MODEL,
        "device": DEVICE
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Faster-Whisper + Auto Speaker Detection API",
        "version": "2.0.0",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
        "features": ["auto_speaker_detection", "silhouette_score"],
        "status": "ready"
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe audio with speaker diarization
    
    Form data:
        - file: Audio file (required)
        - num_speakers: Number of speakers (optional, default: auto-detect)
        - language: Override language (optional, default: from config)
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
    language = request.form.get("language", LANGUAGE)
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    try:
        print(f"🎙️ Transcribing audio: {file.filename}")
        print(f"   Language: {language}, Speakers: {'auto-detect' if num_speakers is None else num_speakers}")
        
        # Transcribe with Faster-Whisper
        segments_generator, info = WHISPER.transcribe(
            input_path,
            language=language,
            word_timestamps=False,
            vad_filter=True,  # Use built-in VAD to skip silence
            vad_parameters={
                "min_silence_duration_ms": 500,
            }
        )
        
        # Convert generator to list
        segments = list(segments_generator)
        
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
        
        # Generate structured meeting notes (summary, highlights, action items, conclusion)
        print("📝 Generating structured meeting notes...")
        meeting_notes = generate_meeting_notes(transcript_text)
        
        summary = meeting_notes.get("summary", "")
        highlights = meeting_notes.get("highlights", {})
        action_items = meeting_notes.get("actionItems", [])
        conclusion = meeting_notes.get("conclusion", "")
        
        # Get unique speakers
        speakers = list(set([s["speaker"] for s in segments_data]))
        speakers.sort()
        
        # Force garbage collection before cleanup
        gc.collect()
        
        # Cleanup temp files with retry for Windows file locking
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                for attempt in range(3):
                    try:
                        os.remove(path)
                        break
                    except PermissionError:
                        time.sleep(0.5)
                        gc.collect()
        
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
            "summary": summary,
            "highlights": highlights,
            "action_items": action_items,
            "conclusion": conclusion,
            "model": WHISPER_MODEL,
            "device": DEVICE,
            "metadata": {
                "duration": round(info.duration, 2),
                "total_speakers": detected_speakers,
                "diarization_mode": "light-heuristic",
                "model": WHISPER_MODEL,
                "device": DEVICE
            }
        })
        
    except Exception as e:
        # Cleanup on error
        try:
            os.remove(input_path)
            if 'wav_path' in locals() and wav_path != input_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass
        
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5005))
    print(f"\n🚀 Starting Faster-Whisper API server on port {port}")
    print(f"   ✨ Auto speaker detection enabled (Silhouette Score)\n")
    app.run(host="0.0.0.0", port=port, debug=False)