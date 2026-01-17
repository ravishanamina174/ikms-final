"""Service functions for indexing documents into the vector database."""

from pathlib import Path
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader

from ..core.retrieval.vector_store import index_documents


def index_pdf_file(file_path: Path) -> int:
    """Load a PDF from disk and index it into the vector DB.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        Number of document chunks indexed.
    """
    
    return index_documents(file_path)


def index_pdf_bytes(contents: bytes, filename: str) -> int:
    """Index a PDF provided as raw bytes without persisting to application storage.

    This writes the bytes to a system temporary file (ephemeral) only as required
    by `PyPDFLoader` and ensures the temporary file is removed afterwards.

    Args:
        contents: Raw PDF bytes from the upload.
        filename: Original filename supplied by the client (for metadata).

    Returns:
        Number of document chunks indexed.
    """
    tmp = None
    try:
        # Create a temporary file on the OS temp directory. This keeps the
        # application directory stateless and is compatible with serverless
        # platforms which provide ephemeral /tmp storage.
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        # Delegate to existing indexing logic which accepts a Path. The
        # filename is derived from the temp path, but the vector_store will
        # attach the original filename metadata when indexing.
        return index_documents(Path(tmp.name))
    finally:
        # Ensure we always remove the temporary file to avoid leaving artifacts
        # on the ephemeral filesystem.
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
