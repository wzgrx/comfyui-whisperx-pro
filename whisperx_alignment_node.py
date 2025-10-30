"""
WhisperX Alignment Node for ComfyUI
Provides accurate word-level timestamps through forced alignment
"""

import os
import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Any


def segment_text(text: str, max_chars: int = 200, language: str = "en") -> List[str]:
    """
    Segment text into smaller chunks based on punctuation and max character limit.

    Args:
        text: Input text to segment
        max_chars: Maximum characters per segment
        language: Language code for language-specific segmentation

    Returns:
        List of text segments
    """
    if not text or not text.strip():
        return []

    # Clean up text
    text = text.strip()

    # Define sentence ending punctuation for different languages
    if language in ["zh", "ja"]:
        # Chinese and Japanese sentence delimiters
        sentence_endings = r'[。！？!?；;]'
    else:
        # English and other Latin-based languages
        sentence_endings = r'[.!?;]'

    # First, try to split by sentence endings
    sentences = re.split(f'({sentence_endings})', text)

    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])

    # Handle last item if it doesn't have punctuation
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])

    # If no sentences were found, treat entire text as one segment
    if not combined_sentences:
        combined_sentences = [text]

    # Further split segments that exceed max_chars
    final_segments = []
    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            final_segments.append(sentence)
        else:
            # Split long sentences by commas, spaces, or character limit
            if language in ["zh", "ja"]:
                # For Chinese/Japanese, split by commas or character count
                sub_parts = re.split(r'[，,、]', sentence)
            else:
                # For English, split by commas, semicolons, or "and"
                sub_parts = re.split(r'[,;]|\s+and\s+', sentence)

            current_segment = ""
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue

                # Check if adding this part exceeds limit
                test_segment = current_segment + (" " if current_segment else "") + part

                if len(test_segment) <= max_chars:
                    current_segment = test_segment
                else:
                    # Save current segment and start new one
                    if current_segment:
                        final_segments.append(current_segment)

                    # If single part is too long, force split by character limit
                    if len(part) > max_chars:
                        for i in range(0, len(part), max_chars):
                            chunk = part[i:i + max_chars].strip()
                            if chunk:
                                final_segments.append(chunk)
                        current_segment = ""
                    else:
                        current_segment = part

            # Add remaining segment
            if current_segment:
                final_segments.append(current_segment)

    return [seg.strip() for seg in final_segments if seg.strip()]


class WhisperXAlignmentNode:
    """
    A ComfyUI node for aligning text transcripts with audio using WhisperX.
    Provides accurate word-level timestamps through phoneme-based forced alignment.
    """

    def __init__(self):
        self.model_a = None
        self.metadata = None
        self.current_language = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to audio file (wav, mp3, etc.)"
                }),
                "input_type": (["plain_text", "json"], {
                    "default": "plain_text"
                }),
                "text_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Input plain text or JSON segments. For plain text: just type your text. For JSON: [{'text': 'Hello', 'start': 0.0, 'end': 1.0}]"
                }),
                "language": (["en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh", "auto"], {
                    "default": "en"
                }),
                "auto_segment": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto segment text",
                    "label_off": "Use as is"
                }),
                "max_chars_per_segment": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "return_char_alignments": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("aligned_segments", "word_segments", "alignment_info")
    FUNCTION = "align_audio_text"
    CATEGORY = "audio/whisperx"

    def align_audio_text(
        self,
        audio_path: str,
        input_type: str,
        text_input: str,
        language: str,
        auto_segment: bool,
        max_chars_per_segment: int,
        return_char_alignments: bool,
        device: str = "auto"
    ) -> Tuple[str, str, str]:
        """
        Align transcription segments with audio to get accurate word-level timestamps.

        Args:
            audio_path: Path to the audio file
            input_type: Type of input (plain_text or json)
            text_input: Text input (plain text or JSON string)
            language: Language code for alignment model
            auto_segment: Whether to automatically segment the text
            max_chars_per_segment: Maximum characters per segment when auto_segment is True
            return_char_alignments: Whether to return character-level alignments
            device: Device to run on (auto, cuda, or cpu)

        Returns:
            Tuple of (aligned_segments_json, word_segments_json, alignment_info_json)
        """
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "WhisperX is not installed. Please install it using:\n"
                "pip install git+https://github.com/m-bain/whisperx.git"
            )

        # Validate audio path
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate text input
        if not text_input or not text_input.strip():
            raise ValueError("Text input cannot be empty")

        # Process input based on input_type
        segments = []
        if input_type == "json":
            # Parse JSON input
            try:
                segments = json.loads(text_input)
                if not isinstance(segments, list):
                    raise ValueError("JSON input must be an array of segment objects")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in text_input: {e}")
        else:
            # Plain text input
            plain_text = text_input.strip()

            if auto_segment:
                # Auto-segment the text
                print(f"Auto-segmenting text with max_chars_per_segment={max_chars_per_segment}")
                text_segments = segment_text(plain_text, max_chars_per_segment, language)
                print(f"Created {len(text_segments)} segments from plain text")

                # Create segment objects without timestamps (WhisperX will generate them)
                segments = [{"text": seg} for seg in text_segments]
            else:
                # Use entire text as one segment
                segments = [{"text": plain_text}]

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load audio
        print(f"Loading audio from: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        # Auto-detect language if needed
        if language == "auto":
            if segments and len(segments) > 0 and "language" in segments[0]:
                language = segments[0]["language"]
            else:
                language = "en"  # Default to English
                print(f"Warning: Could not auto-detect language, defaulting to 'en'")

        # Load alignment model
        print(f"Loading alignment model for language: {language}")
        if self.model_a is None or self.current_language != language:
            self.model_a, self.metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            self.current_language = language
            print(f"Alignment model loaded successfully on device: {device}")

        # Perform alignment
        print(f"Aligning {len(segments)} segments...")
        result = whisperx.align(
            segments,
            self.model_a,
            self.metadata,
            audio,
            device,
            return_char_alignments=return_char_alignments
        )

        # Extract aligned segments
        aligned_segments = result.get("segments", [])

        # Extract word-level segments
        word_segments = []
        for segment in aligned_segments:
            if "words" in segment:
                for word in segment["words"]:
                    word_segments.append(word)

        # Create alignment info
        alignment_info = {
            "language": language,
            "device": device,
            "input_type": input_type,
            "auto_segment": auto_segment,
            "max_chars_per_segment": max_chars_per_segment if auto_segment else None,
            "num_segments": len(aligned_segments),
            "num_words": len(word_segments),
            "return_char_alignments": return_char_alignments,
            "audio_path": audio_path,
        }

        # Add timing statistics if available
        if word_segments:
            times = [(w.get("start", 0), w.get("end", 0)) for w in word_segments if "start" in w and "end" in w]
            if times:
                alignment_info["total_duration"] = max([t[1] for t in times]) if times else 0
                alignment_info["average_word_duration"] = sum([t[1] - t[0] for t in times]) / len(times)

        print(f"Alignment complete! Processed {len(aligned_segments)} segments and {len(word_segments)} words")

        # Return as JSON strings
        return (
            json.dumps(aligned_segments, indent=2, ensure_ascii=False),
            json.dumps(word_segments, indent=2, ensure_ascii=False),
            json.dumps(alignment_info, indent=2, ensure_ascii=False)
        )


class WhisperXTranscribeNode:
    """
    A ComfyUI node for transcribing audio using WhisperX.
    Provides fast transcription with batching support.
    """

    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to audio file"
                }),
                "model_name": (["tiny", "base", "small", "medium", "large-v2", "large-v3"], {
                    "default": "base"
                }),
                "language": (["auto", "en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh"], {
                    "default": "auto"
                }),
                "batch_size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
                "compute_type": (["int8", "float16", "float32"], {
                    "default": "float16"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("segments", "full_text", "transcription_info")
    FUNCTION = "transcribe_audio"
    CATEGORY = "audio/whisperx"

    def transcribe_audio(
        self,
        audio_path: str,
        model_name: str,
        language: str,
        batch_size: int,
        device: str = "auto",
        compute_type: str = "float16"
    ) -> Tuple[str, str, str]:
        """
        Transcribe audio using WhisperX.

        Args:
            audio_path: Path to the audio file
            model_name: Whisper model size
            language: Language code (or 'auto' for auto-detection)
            batch_size: Batch size for faster processing
            device: Device to run on
            compute_type: Computation precision

        Returns:
            Tuple of (segments_json, full_text, transcription_info_json)
        """
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "WhisperX is not installed. Please install it using:\n"
                "pip install git+https://github.com/m-bain/whisperx.git"
            )

        # Validate audio path
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adjust compute_type based on device
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            print("Warning: float16 not supported on CPU, switching to int8")

        # Load model
        print(f"Loading WhisperX model: {model_name}")
        if self.model is None or self.current_model_name != model_name:
            self.model = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type
            )
            self.current_model_name = model_name
            print(f"Model loaded successfully on device: {device}")

        # Load audio
        print(f"Loading audio from: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        # Transcribe
        print(f"Transcribing audio with batch_size={batch_size}...")
        language_param = None if language == "auto" else language
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language_param
        )

        # Extract results
        segments = result.get("segments", [])
        detected_language = result.get("language", "unknown")

        # Create full text
        full_text = " ".join([seg.get("text", "") for seg in segments])

        # Create transcription info
        transcription_info = {
            "model": model_name,
            "language": detected_language,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "num_segments": len(segments),
            "audio_path": audio_path,
        }

        print(f"Transcription complete! Generated {len(segments)} segments")

        # Return as JSON strings
        return (
            json.dumps(segments, indent=2, ensure_ascii=False),
            full_text,
            json.dumps(transcription_info, indent=2, ensure_ascii=False)
        )


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WhisperX Alignment": WhisperXAlignmentNode,
    "WhisperX Transcribe": WhisperXTranscribeNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperX Alignment": "WhisperX Alignment (Text-Audio Align)",
    "WhisperX Transcribe": "WhisperX Transcribe (Audio to Text)",
}
