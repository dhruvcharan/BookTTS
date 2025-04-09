import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, List, Union

import markdown
import nltk
import pyttsx3
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

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


# class MarkdownProcessor2(MarkdownProcessor):
#     def process(self, markdown_text: str) -> str:
#         print("Processing markdown to plain text...")

#         lines = markdown_text.splitlines()

#         # Extract lines from 500 to 800 (indices 499 to 799)
#         lines_to_process = lines[499:800]

#         # Join the lines back together into a single string
#         markdown_text_limited = "\n".join(lines_to_process)

#         # Convert markdown to HTML
#         html = markdown.markdown(markdown_text_limited)

#         # Parse HTML to extract plain text
#         parser = HTMLToTextParser()
#         parser.feed(html)
#         text = parser.get_text()

#         # Clean up the text
#         # Remove extra whitespace
#         text = re.sub(r"\s+", " ", text)

#         return text.strip()


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


class TextChunker:
    @staticmethod
    def chunk_text(text: str, max_tokens: int, tokenizer: Tokenizer) -> List[str]:
        """Split text into chunks that don't exceed max_tokens"""
        print("Chunking text into manageable segments...")
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in tqdm(sentences, desc="Analyzing sentences"):
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

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
        chunks = TextChunker.chunk_text(processed_text, max_tokens, self.tokenizer)

        print(f"Split text into {len(chunks)} chunks for processing")

        temp_files = []
        for i, chunk in enumerate(tqdm(chunks, desc="Generating audio chunks")):
            temp_output = str(Path(output_path).with_suffix(f".part{i}.mp3"))
            temp_files.append(temp_output)

            self.model.generate_audio(chunk, temp_output)

        if len(temp_files) > 1:
            self._combine_audio_files(temp_files, output_path)
            for temp_file in tqdm(temp_files, desc="Cleaning up temp files"):
                Path(temp_file).unlink(missing_ok=True)
        elif len(temp_files) == 1:
            print("Only one chunk created, renaming to final output")
            Path(temp_files[0]).rename(output_path)

        print(f" Conversion complete: {output_path}")
        return output_path

    def _combine_audio_files(self, input_files: List[str], output_file: str) -> None:
        """Combine multiple audio files into one"""
        try:
            from pydub import AudioSegment

            print("Combining audio chunks...")
            combined = AudioSegment.empty()
            for file in tqdm(input_files, desc="Combining audio files"):
                audio = AudioSegment.from_mp3(file)
                combined += audio

            print(f"Exporting final audio to {output_file}")
            combined.export(output_file, format="mp3")
            print(f"Combined {len(input_files)} audio chunks into {output_file}")

        except ImportError:
            print("Warning: pydub not found. Only the first audio chunk will be used.")
            Path(input_files[0]).rename(output_file)


def convert_markdown_to_speech(markdown_path: str, output_path: str = None):
    print(" Starting Markdown to Speech Conversion 🔊")

    processor =MarkdownLibProcessor()

    tokenizer = NLTKTokenizer(max_length=512)

    print("Initializing TTS engine...")
    tts_model = Pyttsx3TTSModel()

    pipeline = TTSPipeline(
        markdown_processor=processor, model=tts_model, tokenizer=tokenizer
    )

    result_path = pipeline.convert(markdown_path, output_path)
    return result_path


if __name__ == "__main__":
    markdown_path = "example.md"
    output_path = convert_markdown_to_speech(markdown_path)
    print(f" Success! Converted: {markdown_path} -> {output_path} ✨")
