from pathlib import PosixPath, Path
from collections import defaultdict
from typing import Optional
from node import Config
from utils import methods

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document


def _read_documents_from_dir(directory: PosixPath,  recursive: Optional[bool] = True) -> list[Document]:
    """
    Fetches and joins the documents from the specified directory.

    Args:
        directory (str): The directory path where the documents are located.

    Returns:
        Document: A single document containing the text of all the documents joined together.

    """
    documents = SimpleDirectoryReader(
        input_dir=directory,
        recursive=recursive,
    ).load_data()
    assert len(documents) > 0, "No documents were found in the directory."
    return documents


def _read_documents_from_dir_using_llama_parse(directory: PosixPath, recursive: Optional[bool] = True) -> list[Document]:
    llama_cloud_secrets = methods.extract_llama_cloud_secrets()
    llama_cloud_api_key = llama_cloud_secrets["llama_cloud_key"]
    parser = LlamaParse(
        api_key=llama_cloud_api_key,
        result_type="text"  # "markdown" and "text" are available
    )
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_dir=directory,
        recursive=recursive,
        file_extractor=file_extractor
    ).load_data()
    assert documents, "No documents were found in the directory."
    return documents


def load_documents(source_path: str, parser_type: str) -> list[Document]:
    """
    Fetches and joins the documents from the specified directory.

    Args:
        directory (str): The directory path where the documents are located.

    Returns:
        Document: A single document containing the text of all the documents joined together.

    """
    path_to_source = Path(source_path)
    assert path_to_source.exists(), "Source directory does not exist."

    documents = None
    if parser_type == "base":
        documents = _read_documents_from_dir(path_to_source, recursive=True)
    elif parser_type == "llamacloud":
        documents = _read_documents_from_dir_using_llama_parse(path_to_source, recursive=True)
    else:
        raise ValueError(f"Invalid parser type: {parser_type}")

    assert documents, "No documents were found in the directory."
    return documents