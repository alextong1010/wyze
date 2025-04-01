from pydub import AudioSegment
import ffmpeg
from pydub.silence import detect_nonsilent
import numpy as np

def extract_audio(video_path, audio_path="extracted_audio.wav"):
    """Extracts audio from an MP4 file and saves it as a WAV file."""
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
        print(f"Audio extracted to: {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")
        return None



def analyze_audio(audio_path, sensitivity=2.0, min_silence_len=500, absolute_threshold=-50):
    """
    Analyzes audio and detects timestamps where volume exceeds:
    - **A dynamic threshold** (mean + std dev of audio loudness)
    - **An absolute threshold** (e.g., -50 dBFS) to always catch very loud sounds
    
    Parameters:
    - `sensitivity`: Adjusts how strict the dynamic threshold is (higher = stricter).
    - `min_silence_len`: Minimum silence duration to consider a segment loud.
    - `absolute_threshold`: Any sound **louder** than this is always detected.
    
    Returns:
    - List of (start, end) timestamps in seconds where loud sounds occur.
    """

    audio = AudioSegment.from_wav(audio_path)

    # Break audio into 50ms chunks and measure volume
    chunk_size = 50  # 50ms per chunk
    loudness_values = [
        audio[i:i+chunk_size].dBFS for i in range(0, len(audio), chunk_size)
    ]

    # Remove extreme low values (-inf) that might come from pure silence
    loudness_values = [x for x in loudness_values if x > -float("inf")]

    if not loudness_values:
        print("No valid audio detected!")
        return []

    # Compute adaptive threshold
    mean_dBFS = np.mean(loudness_values)
    std_dBFS = np.std(loudness_values)
    dynamic_threshold = mean_dBFS + (sensitivity * std_dBFS)

    # Apply absolute threshold correctly
    final_threshold = min(dynamic_threshold, absolute_threshold)  # Use the lowest threshold

    """"
    print(f"\n--- DEBUG INFO ---")
    print(f"Mean dBFS: {mean_dBFS:.2f}, Std Dev: {std_dBFS:.2f}")
    print(f"Computed Dynamic Threshold: {dynamic_threshold:.2f} dBFS")
    print(f"Final Applied Threshold (Min of Dynamic & Absolute): {final_threshold:.2f} dBFS\n")
    """

    # Detect non-silent sections using final threshold
    loud_sections = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=final_threshold)

    # Convert milliseconds to seconds
    loud_times = [(start / 1000, end / 1000) for start, end in loud_sections]

    return loud_times

# Example usage:
video_file = "videos/explosion.mp4"  # Replace with your MP4 file path
audio_file = extract_audio(video_file)

if audio_file:
    loud_sounds = analyze_audio(audio_file)
    print("Loud sounds detected at these times (seconds):")
    # Print each loud section's start and end times
    for start, end in loud_sounds:
        print(f"From {start:.2f}s to {end:.2f}s")
