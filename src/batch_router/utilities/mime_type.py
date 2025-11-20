import filetype as ft
import base64

def get_mime_type(data: str | bytes):
    """
    Get the MIME type of the provided data.

    Args:
        data (str | bytes): The data to inspect. Can be a base64-encoded string or a raw bytes object.

    Returns:
        str: The detected MIME type (e.g., 'image/jpeg', 'audio/wav'). Returns None if detection fails.
    """
    if isinstance(data, str):
        data = base64.b64decode(data)
    return ft.guess_mime(data)
