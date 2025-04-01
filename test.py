import ffmpeg

def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    """Extracts audio from an MP4 video file and saves it as a WAV file."""
    try:
        ffmpeg.input(video_path).output(output_audio_path).run(overwrite_output=True)
        print(f"Audio extracted to: {output_audio_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")
        
# Example usage
video_file = "videos/loud.mp4"  # Replace with your MP4 file path
extract_audio(video_file)
