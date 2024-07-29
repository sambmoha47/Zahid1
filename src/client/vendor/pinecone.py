from loguru import logger
from utils import methods
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore


class PineconeClient:
    def __init__(self) -> None:
        """
        Initialize an instance of PineconeClient.

        Args:
            None

        Returns:
            None
        """
        self._secrets = methods.extract_pinecone_secrets()
        self._pc = None

    def connect(self) -> Pinecone:
        """
        Connect to Azure services.

        Args:
            self_hosted (bool): A boolean indicating whether the client is self-hosted or not.
            use_one_drive (bool, optional): A boolean indicating whether to use OneDrive. Defaults to False.

        Returns:
            None
        """
        api_key = self._secrets.get("pinecone_api_key", None)
        assert api_key is not None, "Pinecone API Key not provided"
        self._pc = Pinecone(api_key=api_key)
        assert self._pc is not None, "Pinecone Client not initialized"

        
    def create_new_vstore(self, index_name: str, dim: int) -> Pinecone:
        self._pc.create_index(
            name=index_name,
            dimension=dim,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
        pinecone_index = self._pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        return vector_store

    def get_existing_vstore(self, index_name: str) -> Pinecone:
        pinecone_index = self._pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        return vector_store

    def list_existing_indexes(self) -> list:
        existing_indexes = self._pc.list_indexes()
        logger.info(f"[PINECONE] Listing existing indexes: {existing_indexes}")
        return existing_indexes

    def index_exists(self, index_name: str) -> bool:
        existing_indexes = self._pc.list_indexes()
        idx_names = [item['name'] for item in existing_indexes]
        return index_name in idx_names

    def delete_index(self, index_name: str) -> None:
        self._pc.delete_index(index_name)