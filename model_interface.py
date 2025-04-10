import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, List, Union
import torch
import soundfile as sf
import markdown
import nltk
import pyttsx3
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from pydub import AudioSegment
import os

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
nltk.download("punkt_tab")


class HTMLToTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return " ".join(self.text)


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text into tokens that can be processed by the TTS model"""
        pass

    @abstractmethod
    def get_max_length(self) -> int:
        """Return the maximum token length supported by this tokenizer"""
        pass


class TTSModel(ABC):
    @abstractmethod
    def generate_audio(self, text: str, output_path: str) -> str:
        """Generate audio from text and save to output_path"""
        pass
    def save_audio(self, audio: Any, output_path: str) -> None:
        """Save the generated audio to a file"""
        pass


class MarkdownProcessor(ABC):
    @abstractmethod
    def process(self, markdown_text: str) -> str:
        """Process markdown into plain text suitable for TTS"""
        pass


class MarkdownLibProcessor(MarkdownProcessor):
    def process(self, markdown_text: str) -> str:
        print("Processing markdown to plain text...")
        # Convert markdown to HTML
        html = markdown.markdown(markdown_text)

        parser = HTMLToTextParser()
        parser.feed(html)
        text = parser.get_text()

        text = re.sub(r"\s+", " ", text)

        return text.strip()


class MarkdownProcessor2(MarkdownProcessor):
    def process(self, markdown_text: str) -> str:
        print("Processing markdown to plain text...")

        lines = markdown_text.splitlines()

        lines_to_process = lines[499:800]

        markdown_text_limited = "\n".join(lines_to_process)

        html = markdown.markdown(markdown_text_limited)

        parser = HTMLToTextParser()
        parser.feed(html)
        text = parser.get_text()

        text = re.sub(r"\s+", " ", text)

        return text.strip()


class NLTKTokenizer(Tokenizer):
    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def get_max_length(self) -> int:
        return self.max_length


class SpaCyTokenizer(Tokenizer):
    def __init__(self, max_length: int = 5000):
        self.max_length = max_length
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.text for token in doc]

    def get_max_length(self) -> int:
        return self.max_length


class Pyttsx3TTSModel(TTSModel):
    def __init__(self, voice_id: str = None, rate: int = 150):
        self.engine = pyttsx3.init()

        if voice_id:
            self.engine.setProperty("voice", voice_id)

        self.engine.setProperty("rate", rate)

    def generate_audio(self, text: str, output_path: str) -> str:
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        return output_path
    def save_audio(self, audio: Any, output_path: str) -> None:
        sf.write(output_path, audio, 44100)

class SileroTTSModel:
    """
    Thin wrapper that keeps *checkpoint* (model_id) and *voice* (voice_id) separate.

    Example
    -------
    >>> tts = SileroTTSModel(language='en',
    ...                      model_id='v3_en',
    ...                      voice_id='en_45')
    >>> tts.generate_audio("Hello, world!", "hello.wav")
    """

    def __init__(
        self,
        language: str = "en",
        model_id: str = "v3_en",
        voice_id: str = "en_0",
        sample_rate: int = 48_000,
        device: str = "cpu",
    ):
        self.language = language
        self.model_id = model_id      
        self.voice_id = voice_id      
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.language,
        )
        self.model.to(self.device)

        # Try to discover the list of voices the checkpoint supports
        self.available_voices = getattr(self.model, "speakers", None)
        if self.available_voices is None:            # older v3 checkpoints
            self.available_voices = self._heuristic_voice_list()

        # Fail fast (or at least warn) if the requested voice is missing
        if self.available_voices and self.voice_id not in self.available_voices:
            warnings.warn(
                f"Voice '{self.voice_id}' not found in checkpoint '{self.model_id}'. "
                f"Falling back to 'random'.  Available voices: {self.available_voices[:10]} …"
            )
            self.voice_id = "random"


    def voices(self):
        """Return the list of voices this checkpoint contains."""
        return self.available_voices

    def generate_audio(self, text: str, output_path: str | Path) -> Path:
        """Synthesize *text* and save it to *output_path* (WAV)."""
        output_path = Path(output_path)
        try:
            audio = self.model.apply_tts(
                text=text, speaker=self.voice_id, sample_rate=self.sample_rate
            )
            self.save_audio(audio, output_path)
        except Exception as e:
            if "too long" in str(e):
                mid = len(text) // 2
                first  = self.generate_audio(text[:mid],  output_path.with_suffix(".a.wav"))
                second = self.generate_audio(text[mid:], output_path.with_suffix(".b.wav"))
                self._concat([first, second], output_path)
                return Path(output_path)
            raise
        
        return output_path
    
    
    def save_audio(self, audio: torch.Tensor, output_path: Path):
        import torchaudio
        temp_wav_path = output_path.with_suffix(".wav")
        torchaudio.save(output_path, audio.unsqueeze(0), self.sample_rate)
        audio_segment = AudioSegment.from_wav(temp_wav_path)
        audio_segment.export(output_path, format="mp3")

        temp_wav_path.unlink(missing_ok=True)

    def _concat(self, parts: List[Path], dst: Path):
        try:
            combined = sum(
                (AudioSegment.from_wav(p) for p in parts if os.path.exists(p)), 
                AudioSegment.empty()
            )
            combined.export(dst, format="wav")
        except Exception as e:
            print(f"Error during concatenation: {e}")


    def _heuristic_voice_list(self):
        """Fallback lists for the common checkpoints that don’t expose .speakers."""
        if self.model_id.startswith("v3_en"):
            return [f"en_{i}" for i in range(118)] + ["random"]
        if self.model_id.startswith("v3_es"):
            return [f"es_{i}" for i in range(3)] + ["random"]
        if self.model_id.startswith("v3_fr"):
            return [f"fr_{i}" for i in range(6)] + ["random"]
        if self.model_id.startswith("v3_de"):
            return ["eva_k", "jonas", "karlsson", "random"]
        return []

class TextChunker:
    MAX_CHARS = 800    
    @staticmethod
    def chunk_text(text: str, tokenizer: Tokenizer) -> List[str]:
        """
        Split *text* into sentences and pack them greedily so that the
        *character* length of each chunk never exceeds MAX_SYMBOLS.
        """
        print("Chunking text for Silero (≤800 chars each)…")
        sentences = sent_tokenize(text)

        chunks, current, length = [], [], 0
        for sent in sentences:
            if length + len(sent) > TextChunker.MAX_CHARS and current:
                chunks.append(" ".join(current))
                current, length = [sent], len(sent)
            else:
                current.append(sent)
                length += len(sent)

        if current:
            chunks.append(" ".join(current))
        return chunks


class TTSPipeline:
    def __init__(
        self,
        markdown_processor: MarkdownProcessor,
        model: TTSModel,
        tokenizer: Tokenizer,
    ):
        self.markdown_processor = markdown_processor
        self.model = model
        self.tokenizer = tokenizer

    def convert(self, markdown_path: str, output_path: str = None) -> str:
        """Convert markdown to audio"""
        print(f"Reading markdown file: {markdown_path}")
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        processed_text = self.markdown_processor.process(markdown_text)

        if not output_path:
            output_path = str(Path(markdown_path).with_suffix(".mp3"))

        max_tokens = self.tokenizer.get_max_length()
        chunks = TextChunker.chunk_text(processed_text, self.tokenizer)

        print(f"Split text into {len(chunks)} chunks for processing")

        temp_files = []
        for i, chunk in enumerate(tqdm(chunks, desc="Generating audio chunks")):
            temp_output = str(Path(output_path).with_suffix(f".part{i}.wav"))
            temp_files.append(temp_output)
            
            self.model.generate_audio(chunk, temp_output)

        if len(temp_files) > 1:
            self._combine_audio_files(temp_files, output_path)
            for temp_file in tqdm(temp_files, desc="Cleaning up temp files"):
                Path(temp_file).unlink(missing_ok=True)
        elif len(temp_files) == 1:
            print("Only one chunk created, converting to mp3")
            self._convert_to_mp3(temp_files[0], output_path)
            Path(temp_files[0]).unlink(missing_ok=True)

        print(f"Conversion complete: {output_path}")
        return output_path

    def _combine_audio_files(self, input_files: List[str], output_file: str) -> None:
        """Combine multiple audio files into one"""
        try:
            from pydub import AudioSegment

            print("Combining audio chunks...")
            combined = AudioSegment.empty()
            for file in tqdm(input_files, desc="Combining audio files"):
                audio = AudioSegment.from_wav(file)
                combined += audio

            print(f"Converting and exporting final audio to {output_file}")
            combined.export(output_file, format="mp3")
            print(f"Combined {len(input_files)} audio chunks into {output_file}")

        except ImportError:
            print("Warning: pydub not found. Only the first audio chunk will be used.")
            self._convert_to_mp3(input_files[0], output_file)

    def _convert_to_mp3(self, wav_file: str, mp3_file: str) -> None:
        """Convert a WAV file to MP3"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_file)
            audio.export(mp3_file, format="mp3")
        except ImportError:
            print("Warning: pydub not found. Cannot convert WAV to MP3.")
            Path(wav_file).rename(mp3_file)


def convert_markdown_to_speech(markdown_path: str, output_path: str = None):
    print(" Starting Markdown to Speech Conversion 🔊")

    processor =MarkdownLibProcessor()

    tokenizer = NLTKTokenizer(max_length=512)

    print("Initializing TTS engine...")
    tts_model = SileroTTSModel()

    pipeline = TTSPipeline(
        markdown_processor=processor, model=tts_model, tokenizer=tokenizer
    )

    result_path = pipeline.convert(markdown_path, output_path)
    return result_path


if __name__ == "__main__":
    markdown_path = "example.md"
    output_path = convert_markdown_to_speech(markdown_path)
    print(f" Success! Converted: {markdown_path} -> {output_path} ✨")
