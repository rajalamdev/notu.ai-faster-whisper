import numpy as np
import wave
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from services.audio_service import extract_audio_features

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

def perform_simple_diarization(audio_path, segments, num_speakers=None):
    """
    Perform simple speaker diarization using audio features + KMeans clustering
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
