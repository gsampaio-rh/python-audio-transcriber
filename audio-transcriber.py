import whisperx
import torch
import os
from pydub import AudioSegment

device = ""
"""Check CUDA availability."""
if torch.cuda.is_available() == 1:
    device = "cuda"
else:
    device = "cpu"

def banner(text):
    """Display a message when the script is working in the background"""
    print(f"# {text} #")

def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device

def get_result(audio_file, duration):
    # Convert MP3 file to WAV format
    sound = AudioSegment.from_mp3(audio_file)
    audio_file_wav = os.path.splitext(audio_file)[0] + '.wav'
    sound.export(audio_file_wav, format="wav")

    # transcribe with original whisper
    banner("Transcribing text")
    model = whisperx.load_model("base", device=check_device())
    result = model.transcribe(audio_file_wav)

    format_result(result, audio_file_wav)

def format_result(result, audio_file):
    # print(result["segments"]) # before alignment

    # load alignment model and metadata
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)

    # Write transcription to file
    output_file = os.path.splitext(audio_file)[0] + '.txt'
    with open(output_file, "w") as f:
        for segment in result_aligned["segments"]:
            f.write(segment["text"] + "\n")

    print(f"Transcription saved to {output_file}")

def main():
    """Main function."""
    audio_file = input("Please enter the path to the MP3 file: ")
    # Get the duration of the audio file
    duration = AudioSegment.from_file(audio_file).duration_seconds
    get_result(audio_file, duration)  # Get audio transcription and translation if needed

if __name__ == "__main__":
    main()
