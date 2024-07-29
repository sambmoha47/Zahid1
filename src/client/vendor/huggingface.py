import torch
from typing import Dict, Union, Optional
from utils import methods

from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

class HuggingFaceClient:
    """
    A client class for interacting with the Hugging Face API.

    Attributes:
        _secrets (dict): A dictionary containing the Hugging Face secrets.
        _self_hosted (bool): A flag indicating whether the client is self-hosted or not.
    """

    def __init__(self):
        self._secrets: dict = methods.extract_hugging_face_secrets()
        self._self_hosted = False

    def _instantiate_hf_embed_model(self, secrets: dict, model_identifier: Optional[str] = "BAAI/bge-m3") -> dict:
        """
        Instantiate a HuggingFaceEmbedding model.

        Args:
            secrets (dict): A dictionary containing secrets.
            model_identifier (str, optional): The identifier of the Hugging Face model. Defaults to "BAAI/bge-large-en-v1.5".

        Returns:
            dict: The instantiated HuggingFaceEmbedding model.
        """
        embed_model = HuggingFaceEmbedding(model_name=model_identifier)
        return embed_model

    def _instantiate_hf_llm_model(self, secrets: dict, model_identifier: str, temperature: Optional[float] = 0.3) -> dict:
        llm_model = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": temperature, "do_sample": False},
            # system_prompt=system_prompt,
            # query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=model_identifier,
            model_name=model_identifier,
            device_map="auto",
            stopping_ids=[50278, 50279, 50277, 1, 0],
            tokenizer_kwargs={"max_length": 4096},
        )
        return llm_model

    def connect(self, self_hosted: Optional[bool] = False):
        """
        Connect to the Hugging Face API.

        Args:
            self_hosted (bool): A flag indicating whether the client is self-hosted or not.
        """
        assert self._secrets is not None, "Hugging Face Secrets not provided"
        self._self_hosted = self_hosted

    def load_model(self, model_category: str, model_prefix: str) -> None:
        """
        Load a Hugging Face model.

        Args:
            model_category (str): The category of the model.
            model_prefix (str): The prefix of the model.

        Returns:
            None
        """
        model = None
        if model_category == "llm":
            model = self._instantiate_hf_llm_model(self._secrets, model_prefix)
        elif model_category == "embedding":
            model = self._instantiate_hf_embed_model(self._secrets, model_prefix)
        else:
            raise ValueError(f"Model category not found: {model_category}")
        assert model is not None, f"Model not found for category: {model_category} and prefix: {model_prefix}"
        return model