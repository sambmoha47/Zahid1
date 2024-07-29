from loguru import logger
from typing import Optional
from utils import methods

from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.gemini import Gemini
from google.oauth2 import service_account

class VertexClient:
    def __init__(self) -> None:
        """
        Initialize an instance of AzureClient.

        Args:
            None

        Returns:
            None
        """
        self._secrets = methods.extract_vertex_secrets()
        # Internal State
        self._self_hosted = False

    def _instantiate_vertex_llm_model(self,
        secrets: dict, model_identifier: Optional[str] = "chat-bison"
    ) -> Vertex:
        """
        Instantiate an AzureOpenAI model for Language Model (LLM) using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "gpt-35-turbo".

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model for Language Model (LLM).
        """
        llm_model = Vertex(
            model=model_identifier, 
            project=secrets.project_id,
            credentials=secrets,
            system_prompt=(
                "You are a virtual assistant designated for the Zahid Group. Your primary function is to field inquiries "
                "pertaining to business operations, human resources, company policies, and other aspects relevant to the organization. "
                "Your responses should accurately address acronyms, abbreviations, and specialized jargon to maintain clear communication. "
                "Be vigilant in understanding the context and details of each inquiry. "
                "Please be aware that some questions may contain acronyms, abbreviations, or specialized terminology. "
                "If a query is unclear or lacks information, do not hesitate to request additional details from the user. "
                "Adhere to the following two critical guidelines:\n"
                "1. Focus exclusively on topics directly related to the Zahid Group and source documents provided.\n"
                "2. Respond to inquiries in the same language in which they are asked to ensure effective communication."
                
            ),
        )
        return llm_model

    def _instantiate_gemini_model(self, secrets: dict, model_identifier: Optional[str] = "gemini-1.5-pro") -> Vertex:
        """
        Instantiate an AzureOpenAI model for Gemini using the provided secrets.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model for Gemini.
        """
        return Gemini(model=model_identifier, project=secrets.project_id, credentials=secrets)

    def _instantiate_vertex_embed_model(self, secrets, model_identifier: Optional[str] = "textembedding-gecko") -> Vertex:
        """
        Instantiate an AzureOpenAI model for Text Embedding using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "textembedding-gecko".

        Returns:
            VertexTextEmbedding: An instance of the AzureOpenAI model for Text Embedding.
        """
        embed_model = VertexTextEmbedding(
            model_name=model_identifier, project=secrets.project_id, credentials=secrets
        )
        return embed_model


    def connect(self, self_hosted: Optional[bool] = False) -> None:
        """
        Connect to Azure services.

        Args:
            self_hosted (bool): A boolean indicating whether the client is self-hosted or not.
            use_one_drive (bool, optional): A boolean indicating whether to use OneDrive. Defaults to False.

        Returns:
            None
        """
        assert self._secrets is not None, "Azure Secrets not provided"
        self._self_hosted = self_hosted

    def load_model(self, model_category: str, model_prefix: str) -> Vertex:
        """
        Load an AzureOpenAI model based on the provided model category and prefix.

        Args:
            model_category (str): The category of the model to be loaded.
            model_prefix (str): The prefix of the model to be loaded.

        Returns:
            AzureOpenAI: An instance of the loaded AzureOpenAI model.

        Raises:
            NotImplementedError: If the model category is not implemented.
        """
        model = None
        if model_category == "llm":
           model = self._instantiate_vertex_llm_model(self._secrets, model_prefix) 
        elif model_category == "embedding":
            model = self._instantiate_vertex_embed_model(self._secrets, model_prefix)
        else:
            raise NotImplementedError(f"Model category {model_category} is not implemented.")

        assert model is not None, "Model not loaded"
        return model