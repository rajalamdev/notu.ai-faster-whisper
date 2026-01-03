import av
import numpy as np
import os
import wave
import subprocess
from typing import Tuple

def get_audio_duration_from_file(file_path: str) -> float:
    """
    Get duration of audio/video file in seconds.
    Works with any format supported by PyAV.
    Falls back to ffprobe if PyAV fails.
    """
    duration = 0
    
    # Method 1: Try PyAV
    try:
        container = av.open(file_path)
        duration = container.duration / av.time_base if container.duration else 0
        # If duration is 0, try to get from audio stream
        if duration == 0 or duration < 1:
            for stream in container.streams.audio:
                if stream.duration:
                    stream_duration = float(stream.duration * stream.time_base)
                    if stream_duration > duration:
                        duration = stream_duration
                        break
            # Also try video stream for video files
            if duration == 0 or duration < 1:
                for stream in container.streams.video:
                    if stream.duration:
                        stream_duration = float(stream.duration * stream.time_base)
                        if stream_duration > duration:
                            duration = stream_duration
                            break
        container.close()
        
        if duration > 1:
            return duration
    except Exception as e:
        print(f"⚠️ PyAV error getting duration: {e}")
    
    # Method 2: Fallback to ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stdout.strip():
            ffprobe_duration = float(result.stdout.strip())
            if ffprobe_duration > 0:
                print(f"✓ Duration from ffprobe: {ffprobe_duration}s")
                return ffprobe_duration
    except Exception as e:
        print(f"⚠️ ffprobe fallback error: {e}")
    
    # Return whatever we got from PyAV, even if 0
    return duration


def get_wav_info(wav_path: str) -> Tuple[int, int, float]:
    """
    Get WAV file info: sample_rate, num_channels, duration
    """
    try:
        with wave.open(wav_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.getnframes()
            duration = frames / sample_rate
            return sample_rate, channels, duration
    except Exception as e:
        print(f"⚠️ Error reading WAV info: {e}")
        return 16000, 1, 0


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


def convert_to_wav_ffmpeg(input_path, output_path):
    """
    Convert audio to WAV using ffmpeg subprocess.
    More robust for handling partial/malformed webm chunks from MediaRecorder.
    Falls back to PyAV if ffmpeg is not available.
    """
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                output_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            print(f"⚠️ FFmpeg conversion failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            # Fallback to PyAV
            return convert_to_wav(input_path, output_path)
            
    except FileNotFoundError:
        print("⚠️ FFmpeg not found in PATH, trying PyAV...")
        return convert_to_wav(input_path, output_path)
    except subprocess.TimeoutExpired:
        print("⚠️ FFmpeg conversion timed out, trying PyAV...")
        return convert_to_wav(input_path, output_path)
    except Exception as e:
        print(f"⚠️ FFmpeg conversion error: {e}, trying PyAV...")
        return convert_to_wav(input_path, output_path)

try:
    from python_speech_features import mfcc
except ImportError:
    print("⚠️ python_speech_features not found. Installing it is recommended for better diarization.")
    mfcc = None

def extract_audio_features(audio_data, sample_rate=16000):
    """
    Extract audio features for speaker clustering.
    Uses MFCC (Mel-Frequency Cepstral Coefficients) which captures voice timbre.
    """
    if len(audio_data) == 0:
        return np.zeros(14) if mfcc else np.zeros(3)

    if mfcc is not None:
        try:
            # Extract MFCCs (13 coefficients)
            # nfft 512 is enough for 25ms window at 16khz (400 samples)
            mfcc_feat = mfcc(audio_data, samplerate=sample_rate, numcep=13, nfilt=26, nfft=512)
            
            # Take the mean of MFCC vectors across time to get a single embedding for the segment
            # This represents the "average timbre" of the speaker in this segment
            feature_vector = np.mean(mfcc_feat, axis=0) # shape (13,)
            
            # Add Log Energy for robustness
            energy = np.sum(audio_data ** 2) / len(audio_data)
            log_energy = np.log(energy + 1e-10)
            
            return np.concatenate((feature_vector, [log_energy]))
        except Exception as e:
            print(f"⚠️ MFCC extraction error: {e}")
            # Fallback to simple stats
            pass
    
    # Simple features: mean, std, energy
    mean_val = np.mean(audio_data)
    std_val = np.std(audio_data)
    energy = np.sum(audio_data ** 2) / len(audio_data)
    
    return np.array([mean_val, std_val, energy])
