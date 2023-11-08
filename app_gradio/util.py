"""Utility functions for app_gradio module."""
import base64
from io import BytesIO


def encode_b64_image(image, format="png"):
    """Encode a PIL image as a base64 string."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format=format)  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")
    return encoded_image
