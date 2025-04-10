from abc import ABC, abstractmethod
from pathlib import Path

from markitdown import MarkItDown


class FileConverter(ABC):
    @abstractmethod
    def convert_to_markdown(self, filepath: str) -> str:
        pass


class MarkitdownConverter(FileConverter):
    def convert_to_markdown(self, filepath: str) -> str:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        md = MarkItDown()
        result = md.convert(filepath)
        return result.text_content  


class MarkdownConversionManager:
    def __init__(self, converter: FileConverter):
        self.converter = converter

    def convert(self, filepath: str, output_path: str = None) -> str:
        markdown_content = self.converter.convert_to_markdown(filepath)
        if not output_path:
            output_path = Path(filepath).with_suffix(".md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return output_path

