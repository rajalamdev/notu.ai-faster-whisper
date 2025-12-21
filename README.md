# Faster-Whisper + Resemblyzer Diarization API

API transcription audio dengan speaker diarization menggunakan **Faster-Whisper** (4x lebih cepat dari OpenAI Whisper) dan **Resemblyzer** untuk identifikasi speaker.

## ✨ Features

- ⚡ **4x lebih cepat** dari OpenAI Whisper dengan akurasi yang sama
- 🎙️ **Speaker Diarization** menggunakan Resemblyzer + Spectral Clustering
- 🇮🇩 **Optimized untuk Bahasa Indonesia**
- 💾 **Hemat RAM** (~1.7GB) - cocok untuk CPU dengan 8GB RAM
- 🔇 **Built-in VAD** untuk skip silence
- 📝 **Auto Summary** menggunakan OpenRouter AI
- 🚀 **Fast INT8 Quantization** untuk CPU inference

## 🔧 System Requirements

- **Python**: 3.9 atau lebih tinggi
- **RAM**: Minimum 4GB, recommended 8GB
- **FFmpeg**: Diperlukan untuk konversi audio (optional, faster-whisper bisa load audio langsung)

### Install FFmpeg (Optional)

**Windows:**
```bash
# Menggunakan Chocolatey
choco install ffmpeg

# Atau download dari: https://ffmpeg.org/download.html
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## 🚀 Quick Start (Local)

### 1. Clone & Navigate
```bash
cd faster-whisper
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: First run akan download model Faster-Whisper (~140MB untuk base model) dan Resemblyzer (~17MB). Pastikan ada koneksi internet.

### 4. Configure Environment
Edit `.env` file sesuai kebutuhan:
```env
OPENROUTER_API_KEY=your-api-key-here
WHISPER_MODEL=base        # tiny, base, small, medium, large
DEVICE=cpu                # cpu atau cuda
COMPUTE_TYPE=int8         # int8, float16, float32
LANGUAGE=id               # id (Indonesia), en (English), dll
PORT=5005
```

### 5. Run Server
```bash
source venv/Scripts/activate && python app.py
```

Server akan berjalan di `http://localhost:5005`

## 📡 API Usage

### Health Check
```bash
curl http://localhost:5005/
```

**Response:**
```json
{
  "service": "Faster-Whisper + Resemblyzer Diarization API",
  "version": "1.0.0",
  "model": "base",
  "device": "cpu",
  "compute_type": "int8",
  "language": "id",
  "status": "ready"
}
```

### Transcribe Audio
```bash
curl -X POST http://localhost:5005/transcribe \
  -F "file=@audio.mp3" \
  -F "num_speakers=2"
```

**Parameters:**
- `file` (required): Audio file (mp3, wav, m4a, dll)
- `num_speakers` (optional): Fixed number of speakers. Jika tidak diisi, akan auto-detect
- `min_speakers` (optional): Minimum speakers untuk auto-detect (default: 2)
- `max_speakers` (optional): Maximum speakers untuk auto-detect (default: 4)
- `language` (optional): Override language detection (default: dari .env)

**Response:**
```json
{
  "language": "id",
  "language_probability": 0.9876,
  "duration": 123.45,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Selamat pagi semuanya",
      "speaker": "SPEAKER_0"
    },
    {
      "start": 4.2,
      "end": 7.8,
      "text": "Terima kasih sudah hadir",
      "speaker": "SPEAKER_1"
    }
  ],
  "speakers": ["SPEAKER_0", "SPEAKER_1"],
  "transcript": "Selamat pagi semuanya Terima kasih sudah hadir...",
  "summary": "Ringkasan rapat:\n1. Pembukaan...",
  "model": "base",
  "device": "cpu"
}
```

### Example using Python requests
```python
import requests

url = "http://localhost:5005/transcribe"
files = {"file": open("meeting.mp3", "rb")}
data = {"num_speakers": 3}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Detected {len(result['speakers'])} speakers")
for segment in result['segments']:
    print(f"[{segment['speaker']}] {segment['text']}")
```

### Example using JavaScript/fetch
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('num_speakers', '2');

const response = await fetch('http://localhost:5005/transcribe', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Transcript:', result.transcript);
console.log('Summary:', result.summary);
```

## ⚙️ Configuration

### Model Selection
Pilih model sesuai kebutuhan speed vs accuracy:

| Model | Size | RAM | Speed (CPU) | Akurasi |
|-------|------|-----|-------------|---------|
| tiny | 39M | ~1GB | ~10x | ⭐⭐⭐ |
| **base** | 74M | ~1GB | ~7x | ⭐⭐⭐⭐ (recommended) |
| small | 244M | ~2GB | ~4x | ⭐⭐⭐⭐⭐ |
| medium | 769M | ~5GB | ~2x | ⭐⭐⭐⭐⭐ |

**Recommendation untuk 8GB RAM**: Gunakan `base` atau `small`

### Language Support
Faster-Whisper support 99+ bahasa. Set `LANGUAGE` di `.env`:
- `id` - Indonesian
- `en` - English
- `zh` - Chinese
- `ja` - Japanese
- dll (lihat [Whisper language codes](https://github.com/openai/whisper#available-models-and-languages))

## 🎯 Performance Tips

### Untuk CPU 8GB RAM:
1. ✅ Gunakan `WHISPER_MODEL=base` dan `COMPUTE_TYPE=int8`
2. ✅ Set `LANGUAGE=id` untuk skip detection (lebih cepat)
3. ✅ Jangan jalankan aplikasi lain yang memory-intensive

### Expected Processing Speed:
- Audio 1 menit → ~30-60 detik processing (base + int8 + CPU)
- Audio 10 menit → ~5-10 menit processing

## 🐛 Troubleshooting

### Error: "No module named 'faster_whisper'"
```bash
pip install --upgrade faster-whisper
```

### Error: FFmpeg not found
Install FFmpeg atau biarkan faster-whisper load audio langsung (built-in PyAV)

### Out of Memory
Gunakan model lebih kecil:
```env
WHISPER_MODEL=tiny
```

### Diarization tidak akurat
1. Pastikan audio quality bagus (minimal 16kHz)
2. Set `num_speakers` secara manual jika tahu jumlah speaker
3. Adjust `min_speakers` dan `max_speakers` range

## 📦 Production Deployment

Untuk deployment production, lihat:
- `Dockerfile` - Container image
- `docker-compose.yml` - Docker orchestration
- Atau deploy ke cloud service (Railway, Render, Fly.io, dll)

## 🔐 Security Notes

- **Jangan commit** `.env` file ke Git (sudah ada di `.gitignore`)
- Ganti `OPENROUTER_API_KEY` dengan key Anda sendiri
- Untuk production, gunakan environment variables dari hosting platform

## 📝 License

MIT License - Feel free to use for personal and commercial projects

## 🤝 Contributing

Contributions welcome! Open an issue atau submit PR.

## 📚 Tech Stack

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - ASR (Speech-to-Text)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Speaker Embeddings
- [Spectral Cluster](https://github.com/wq2012/SpectralCluster) - Speaker Clustering
- [Flask](https://flask.palletsprojects.com/) - Web Framework
- [OpenRouter](https://openrouter.ai/) - AI Summary Generation
