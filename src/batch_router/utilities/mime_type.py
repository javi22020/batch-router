import filetype as ft
import base64

def get_mime_type(data: str | bytes):
    """Get the mime type of the data.
    Args:
        data: The data to get the mime type of. Can be a base64-encoded string or a bytes object.
    Returns:
        The mime type of the data.
    """
    if isinstance(data, str):
        data = base64.b64decode(data)
    return ft.guess_mime(data)
