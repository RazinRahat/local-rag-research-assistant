import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize extracted text without destroying document structure."""

    text = unicodedata.normalize("NFC", text)

    text = text.replace("\x00", "")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]

    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
