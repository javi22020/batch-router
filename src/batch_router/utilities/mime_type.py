import magic
import base64

def get_mime_type(data: str) -> str:
    """Get the mime type of the data.
    Args:
        data: The data to get the mime type of. Can be a base64-encoded string or a bytes object.
    Returns:
        The mime type of the data.
    """
    if isinstance(data, str):
        data = base64.b64decode(data)
    return magic.from_buffer(base64.b64decode(data))
