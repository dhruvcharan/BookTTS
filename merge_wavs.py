import glob
import os
import re
from pydub import AudioSegment
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

current_dir = os.path.dirname(os.path.abspath(__file__))
search_pattern = os.path.join(current_dir, 'example.part*.wav')

# Find all WAV files matching the pattern
wav_files = glob.glob(search_pattern)
print(f"Searching in directory: {current_dir}")
print(f"Using pattern: {search_pattern}")
print(f"Found {len(wav_files)} files")
print("Files found:", wav_files)

wav_files.sort(key=natural_sort_key)

final_audio = AudioSegment.empty()

for wav_file in tqdm(wav_files):
    audio = AudioSegment.from_wav(wav_file)
    mp3_file = wav_file.replace('.wav', '.mp3')
    audio.export(mp3_file, format='mp3')
    
    final_audio += AudioSegment.from_mp3(mp3_file)

final_audio.export("merged_output.mp3", format="mp3")
