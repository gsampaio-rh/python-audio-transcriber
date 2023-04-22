import whisperx
import time
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
    print(f"##### {text} #####")

def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device

def get_model():
    """Prompt the user to select a Whisper model."""
    models = ['tiny', 'base', 'small', 'medium', 'large']
    banner("Please select a Whisper model:")
    for i, model in enumerate(models):
        print(f"{i+1}: {model}")
    choice = input("Enter the number of the model you wish to use: ")
    while not choice.isdigit() or int(choice) not in range(1, len(models)+1):
        choice = input("Invalid choice. Please enter the number of the model you wish to use: ")
    return models[int(choice)-1]

def print_time(total_time):
    """Prints time in seconds or minutes"""
    if total_time >= 60:
        total_time = total_time / 60
        print(f"{total_time:.2f} minutes")
    else:
        print(f"{total_time:.2f} seconds")

def get_audio_info(audio_file):
    """Get information about the audio file."""
    audio = AudioSegment.from_file(audio_file)
    duration = audio.duration_seconds
    size = os.path.getsize(audio_file)
    created_date = time.ctime(os.path.getctime(audio_file))
    return audio, duration, size, created_date

def print_audio_info(audio_file):
    """Print information about the audio file."""
    audio, duration, size, created_date = get_audio_info(audio_file)
    banner("Audio file information:")
    print(f"  Name: {os.path.basename(audio_file)}")
    print("  Duration: ")
    print_time(duration)
    print(f"  Size: {size / 1024:.2f} KB")
    print(f"  Created date: {created_date}\n")

def print_date(timestamp):
    """Prints a timestamp in a human-readable format."""
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return date_string

def estimate_transcription_time(audio_file, model):
    """Estimate the time taken to transcribe the audio file."""
    audio, duration, _, _ = get_audio_info(audio_file)
    banner("Transcribing 120 seconds to estimate")
    audio_duration = min(duration, 120)  # transcribe at most 120 seconds of audio
    print("Audio Estimation Sample Duration: ")
    print_time(audio_duration)
    audio_segment = audio[:audio_duration * 1000]
    audio_segment.export('audio_segment.wav', format="wav")
    start_time = time.time()
    model.transcribe('audio_segment.wav')
    elapsed_time_30= time.time() - start_time
    print("Total time taken to transcribe 30 seconds of audio: ")
    print_time(elapsed_time_30)
    tps = elapsed_time_30/audio_duration
    print("Transcription per second: ")
    print_time(tps)
    total_estimated_time = tps * duration
    print("Estimated time to transcribe the full audio: ")
    print_time(total_estimated_time)
    return total_estimated_time, model

def transcribe_audio(audio_file, model):
    """Transcribe an audio file."""
    
    banner("Transcribing text")
    start_time = time.time()
    print(f"Start transcription time: {print_date(start_time)}")
    result = model.transcribe(audio_file)
    end_time = time.time()
    print(f"End transcription time: {print_date(end_time)}")
    elapsed_time = end_time - start_time
    print("Total time taken to transcribe the full audio: ")
    print_time(elapsed_time)
    banner("Transcription complete")

    return result
    
def format_result(result, audio_file):
    banner("Format result")
    start_time = time.time()
    print(f"Start format time: {print_date(start_time)}")
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

    end_time = time.time()
    print(f"End format time: {print_date(end_time)}")
    elapsed_time = end_time - start_time
    print("Total time taken to format the full text: ")
    print_time(elapsed_time)

    banner(f"Transcription saved to {output_file}")

def main():
    """Main function."""
    audio_file = input("Please enter the path to the MP3 file: ")

    # Get the duration of the audio file
    audio = AudioSegment.from_file(audio_file)
    duration = audio.duration_seconds
    
    print_audio_info(audio_file)
    
    model_name = get_model()
    model = whisperx.load_model(model_name, device=check_device())

    total_estimated_time, model = estimate_transcription_time(audio_file, model)

    # Transcribe the full audio file
    result = transcribe_audio(audio_file, model)
    
    # Format the result
    format_result(result, audio_file)

if __name__ == "__main__":
    main()
