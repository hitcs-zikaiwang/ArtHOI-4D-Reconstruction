from openai import OpenAI
import time
import base64
import requests
import dashscope
import os
from natsort import natsorted
from dashscope import MultiModalConversation
import json
import re
from collections import defaultdict
import multiprocessing
try:
    from google import genai  # google-genai SDK
    from google.genai import types
except ImportError:  # Fall back gracefully when the SDK is not installed.
    genai = None
    types = None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

class GeminiSingle:
    """Gemini single-image caller, similar to QwenSingle: request_with_image(prompt, image_path)."""
    def __init__(self, api_key: str | None = None, model: str = 'gemini-2.5-flash'):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'CHANGE_ME_GEMINI_KEY')
        self.model = model
        if genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self.client = None
                self._init_error = f"Gemini client init failed: {e}"
        else:
            self.client = None
            self._init_error = "google-genai library not installed"

    @staticmethod
    def _infer_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            return 'image/jpeg'
        if ext == '.png':
            return 'image/png'
        if ext == '.webp':
            return 'image/webp'
        return 'application/octet-stream'

    def request_with_image(self, prompt: str, image_path: str) -> str:
        if self.client is None:
            return f"[GeminiSingle Init Error] {getattr(self, '_init_error', 'no client')}"
        if not os.path.isfile(image_path):
            return f"[GeminiSingle] image not found: {image_path}"
        mime = self._infer_mime(image_path)
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    prompt
                ]
            )
            return getattr(resp, 'text', '') or ''
        except Exception as e:
            return f"[GeminiSingle Error] {e}"


class GeminiMulti:
    """Gemini multi-image caller: upload the first image and embed the rest as bytes."""
    def __init__(self, api_key: str | None = None, model: str = 'gemini-2.5-flash'):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'CHANGE_ME_GEMINI_KEY')
        self.model = model
        if genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self.client = None
                self._init_error = f"Gemini client init failed: {e}"
        else:
            self.client = None
            self._init_error = "google-genai library not installed"

    @staticmethod
    def _infer_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            return 'image/jpeg'
        if ext == '.png':
            return 'image/png'
        if ext == '.webp':
            return 'image/webp'
        return 'application/octet-stream'

    def request_with_images(self, prompt: str, image_paths: list[str]) -> str:
        if self.client is None:
            return f"[GeminiMulti Init Error] {getattr(self, '_init_error', 'no client')}"
        if not image_paths:
            return "[GeminiMulti] empty image_paths"
        # Filter out missing files.
        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if not valid_paths:
            return "[GeminiMulti] no valid images"
        parts = [prompt]
        try:
            # Try uploading the first image, following the official example.
            first_path = valid_paths[0]
            try:
                uploaded = self.client.files.upload(file=first_path)
                parts.append(uploaded)
            except Exception as e:
                # Fall back to embedded bytes.
                with open(first_path, 'rb') as f:
                    b = f.read()
                parts.append(types.Part.from_bytes(data=b, mime_type=self._infer_mime(first_path)))
            # Embed the remaining images as bytes.
            for p in valid_paths[1:]:
                try:
                    with open(p, 'rb') as f:
                        b = f.read()
                    parts.append(types.Part.from_bytes(data=b, mime_type=self._infer_mime(p)))
                except Exception as e:
                    parts.append(types.Part.from_bytes(data=b'', mime_type='application/octet-stream'))
            resp = self.client.models.generate_content(
                model=self.model,
                contents=parts
            )
            return getattr(resp, 'text', '') or ''
        except Exception as e:
            return f"[GeminiMulti Error] {e}"
        
class GeminiPerspective:
    """Gemini perspective classification with multiple images.
    The interface is similar to QwenPerspective and uses Gemini multimodal input.
    request_with_images(prompt, image_paths)
    - Try uploading the first image to obtain a file reference.
    - Embed the remaining images as bytes.
    - Use prompt as the first content part.
    Return the model text result or an error string.
    """
    def __init__(self, api_key: str | None = None, model: str = 'gemini-2.5-flash'):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'CHANGE_ME_GEMINI_KEY')
        self.model = model
        if genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self.client = None
                self._init_error = f"Gemini client init failed: {e}"
        else:
            self.client = None
            self._init_error = "google-genai library not installed"

    @staticmethod
    def _infer_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            return 'image/jpeg'
        if ext == '.png':
            return 'image/png'
        if ext == '.webp':
            return 'image/webp'
        return 'application/octet-stream'

    def request_with_images(self, prompt: str, image_paths: list[str]) -> str:
        if self.client is None:
            return f"[GeminiPerspective Init Error] {getattr(self, '_init_error', 'no client')}"
        if not image_paths:
            return "[GeminiPerspective] empty image_paths"
        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if not valid_paths:
            return "[GeminiPerspective] no valid images"
        parts = [prompt]
        try:
            first_path = valid_paths[0]
            try:
                uploaded = self.client.files.upload(file=first_path)
                parts.append(uploaded)
            except Exception:
                with open(first_path, 'rb') as f:
                    b = f.read()
                parts.append(types.Part.from_bytes(data=b, mime_type=self._infer_mime(first_path)))
            for p in valid_paths[1:]:
                try:
                    with open(p, 'rb') as f:
                        b = f.read()
                    parts.append(types.Part.from_bytes(data=b, mime_type=self._infer_mime(p)))
                except Exception:
                    parts.append(types.Part.from_bytes(data=b'', mime_type='application/octet-stream'))
            resp = self.client.models.generate_content(
                model=self.model,
                contents=parts
            )
            return getattr(resp, 'text', '') or ''
        except Exception as e:
            return f"[GeminiPerspective Error] {e}"

