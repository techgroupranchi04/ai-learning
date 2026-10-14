
from pathlib import Path


# def clean_text(text):
#     """Remove extra whitespace and blank lines."""
#     lines = [line.strip() for line in text.splitlines()]
#     lines = [line for line in lines if line]
#     return "\n".join(lines)

def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


def chunk_text(text, chunk_size=100, overlap=20):
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


# 1. Read the document
raw_text = Path("sample_document.txt").read_text(encoding="utf-8")

# 2. Preprocess the document
cleaned_text = raw_text  # Skipping the cleaning step as clean_text is commented out

# 3. Split the text into chunks
chunks = chunk_text(cleaned_text, chunk_size=100, overlap=20)

# 4. Display the results
print("CLEANED DOCUMENT:")
print(cleaned_text)

print("\nTOTAL CHARACTERS:", len(cleaned_text))
print("TOTAL CHUNKS:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ({len(chunk)} characters) ---")
    print(chunk)
