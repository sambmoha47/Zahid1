from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import Optional
from google.oauth2 import service_account
import subprocess
import fitz
import os


def find_file(directory: str, filename: str) -> str:
    """
    Finds a file with the given filename in the specified directory.

    Args:
        directory (str): The directory to search for the file.
        filename (str): The name of the file to find.

    Returns:
        str: The path of the found file, or None if the file is not found.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                return os.path.join(root, file)
    return None

def extract_hugging_face_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.huggingface", override=True)
    hugging_face_api_key = os.environ["HUGGING_FACE_API_KEY"]

    secrets = {
        "huggin_face_api_key": hugging_face_api_key,
    }
    return secrets


def extract_azure_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.azure", override=True)
    azure_api_endpoint_url = os.environ["AZURE_API_ENDPOINT_URL"]
    azure_api_type = os.environ["AZURE_API_TYPE"]
    azure_api_version = os.environ["AZURE_API_VERSION"]
    azure_api_key = os.environ["AZURE_API_KEY"]
    azure_embedding_deployement_name = os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
    azure_llm_deployement_name = os.environ["AZURE_LLM_DEPLOYMENT_NAME"]

    secrets = {
        "azure_api_endpoint_url": azure_api_endpoint_url,
        "azure_api_type": azure_api_type,
        "azure_api_version": azure_api_version,
        "azure_api_key": azure_api_key,
        "azure_embedding_deployment_name": azure_embedding_deployement_name,
        "azure_llm_deployment_name": azure_llm_deployement_name,
    }

    return secrets

def extract_openai_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.openai", override=True)
    openai_api_key = os.environ["OPENAI_API_KEY"]

    secrets = {
        "openai_api_key": openai_api_key,
    }
    return secrets

def extract_grok_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.grok", override=True)
    grok_api_key = os.environ["GROK_API_KEY"]

    secrets = {
        "grok_api_key": grok_api_key,
    }
    return secrets

def extract_vertex_secrets() -> dict:
    filename = Path() / "secrets" / "vertex.json"
    credentials: service_account.Credentials = (
    service_account.Credentials.from_service_account_file(filename)
    )
    return credentials


def extract_neo4j_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(find_dotenv(), override=True)
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    url = os.environ["NEO4J_ENDPOINT_URI"]
    aura_instance_id = os.environ["AURA_INSTANCE_ID"]
    aura_instance_name = os.environ["AURA_INSTANCE_NAME"]

    secrets = {
        "username": username,
        "password": password,
        "url": url,
        "aura_instance_id": aura_instance_id,
        "aura_instance_name": aura_instance_name,
    }
    return secrets

def extract_llama_cloud_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.llamacloud", override=True)
    llama_cloud_key = os.environ["LLAMACLOUD_API_KEY"]
    secrets = {
        "llama_cloud_key": llama_cloud_key,
    }
    return secrets

def extract_pinecone_secrets() -> dict:
    """
    Extracts secrets from environment variables and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the extracted secrets.
    """
    _ = load_dotenv(dotenv_path="secrets/.pinecone", override=True)
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    secrets = {
        "pinecone_api_key": pinecone_api_key,
    }
    return secrets


def convert_ppt_to_pdf(source_folder: str) -> None:
    """
    Convert all PowerPoint files (.ppt and .pptx) in the given source folder
    and its subdirectories to PDF format using LibreOffice.

    Args:
        source_folder (str): The path to the folder containing PowerPoint files.
    """
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith((".ppt", ".pptx")):
                ppt_path = os.path.join(root, file)

                # Convert PPT to PDF using LibreOffice
                subprocess.run(
                    [
                        "libreoffice",
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        root,
                        ppt_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )


def highlight_search_terms(
    pdf_document_path: str,
    page: int,
    search_term: str,
    highlight_color: Optional[tuple] = (1, 0, 0),
    output_path: Optional[str] = "./data/highlight/highlighted.pdf",
):
    """
    Highlights the search terms in the given PDF document.

    Args:gg
        pdf_document (fitz.Document): The PDF document object.
        page (fitz.Page): The page object where the search will be performed, NOTE: 0-based index.
        search_term (str): The search term to be highlighted.
        highlight_color (tuple, optional): The color of the highlight. Defaults to (1, 0, 0) (red).
        output_path (str, optional): The path where the modified PDF will be saved. Defaults to "../docs/highlighted.pdf".
    """
    search_words = search_term.split()
    pdf_document = fitz.open(pdf_document_path)
    pdf_document_at_selected_page = pdf_document[page - 1]

    # Search for the text and highlight it
    for i in range(len(search_words) - 1):
        two_word_term = " ".join(search_words[i : i + 2])
        text_instances = pdf_document_at_selected_page.search_for(two_word_term)
        for inst in text_instances:
            highlight = pdf_document_at_selected_page.add_highlight_annot(inst)
            highlight.set_colors(stroke=highlight_color)
            highlight.update()

    # Save the modified PDF
    pdf_document.save(output_path)
    return pdf_document, output_path


def fetch_response_results(response: dict, preprocess: bool) -> dict:
    """
    Fetches the relevant information from the response object and returns it as a dictionary.

    Args:
        response (dict): The response object containing the necessary information.
        preprocess (bool): A flag indicating whether preprocessing should be applied to the text.

    Returns:
        dict: A dictionary containing the fetched information, including the text, pages, file paths, and references.
    """
    text = response.response
    source_nodes = response.source_nodes
    pages = [
        n.metadata.get("page_label")
        for n in source_nodes
        if n.metadata.get("page_label") is not None
    ]
    file_paths = [
        n.metadata.get("file_path")
        for n in source_nodes
        if n.metadata.get("file_path") is not None
    ]

    references = [n.text for n in source_nodes]

    response_info = {
        "text": text,
        "pages": pages,
        "file_paths": file_paths,
        "references": references,
    }
    return response_info