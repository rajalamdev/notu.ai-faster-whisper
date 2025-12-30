import av
import numpy as np
import os

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
