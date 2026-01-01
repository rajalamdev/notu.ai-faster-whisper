from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from routes.transcription_routes import transcription_bp
from routes.chat_routes import chat_bp
from services.transcription_service import TranscriptionService

load_dotenv()

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(transcription_bp)
app.register_blueprint(chat_bp)

# Preload Model at Startup
try:
    TranscriptionService.get_model()
except Exception as e:
    print(f"⚠️ Warning: Failed to preload model: {e}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "faster-whisper-modular"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Notu.AI Transcription API", 
        "version": "2.1.0-modular",
        "features": ["auto_speaker_detection", "mfcc_ready"]
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5005))
    print(f"\n🚀 Starting Faster-Whisper API server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)