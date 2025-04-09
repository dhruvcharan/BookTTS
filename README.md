# TTSConversion

## Overview

The `TTSConversion` project is designed to facilitate the conversion of various file types into Markdown (`.md`) format and subsequently convert Markdown files into Text-to-Speech (TTS) audio output. 

### Key Features:
1. **Filetype to Markdown Conversion**:
   - Supports multiple file formats (e.g., `.txt`, `.docx`, `.pdf`).
   - Extracts content and formats it into a structured Markdown file.

2. **Markdown to TTS Conversion**:
   - Reads the content of Markdown files.
   - Converts the text into audio using TTS libraries.
   - Outputs audio files in common formats (e.g., `.mp3`, `.wav`).

### Workflow:
1. Input a file in a supported format.
2. Convert the file into a Markdown file.
3. Use the Markdown file to generate a TTS audio output.

## Instructions to Run Scripts Using `uv`

1. Ensure `uv` is installed on your system. If not, install it using:
   ```bash
   pip install uv
   ```

2. Navigate to the project directory:
   ```bash
   cd /Users/dhruvcharan/code/TTSConversion
   ```

3. Run the desired script using `uv`. For example:
   ```bash
   uv run script_name.py
   ```

   Replace `script_name.py` with the name of the script you want to execute.

4. For additional options or help, use:
   ```bash
   uv --help
   ```

## Additional Notes

- Ensure all dependencies are installed before running the scripts.
- Refer to the project documentation for specific script details.
