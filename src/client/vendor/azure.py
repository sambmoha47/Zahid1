from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from loguru import logger
from typing import Optional
from utils import methods


class AzureClient:
    def __init__(self) -> None:
        """
        Initialize an instance of AzureClient.

        Args:
            None

        Returns:
            None
        """
        self._secrets = methods.extract_azure_secrets()
        # Internal State
        self._self_hosted = False

    def _instantiate_azure_llm_model(self,
        secrets: dict, model_identifier: Optional[str] = "gpt-35-turbo"
    ) -> AzureOpenAI:
        """
        Instantiate an AzureOpenAI model for Language Model (LLM) using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "gpt-35-turbo".

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model for Language Model (LLM).
        """
        azure_llm_model = AzureOpenAI(
            model=model_identifier,
            temperature=0.3,
            deployment_name=secrets.get("azure_llm_deployment_name"),
            api_key=secrets.get("azure_api_key"),
            azure_endpoint=secrets.get("azure_api_endpoint_url"),
            api_version=secrets.get("azure_api_version")
        )
        return azure_llm_model

    def _instantiate_azure_embedding_model(self,
        secrets: dict, model_identifier: Optional[str] = "text-embedding-ada-002", temperature: Optional[float] = 0.3
    ) -> AzureOpenAIEmbedding:
        """
        Instantiate an AzureOpenAI model for Embedding using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "text-embedding-ada-002".

        Returns:
            AzureOpenAIEmbedding: An instance of the AzureOpenAI model for Embedding.
        """
        azure_embedding_model = AzureOpenAIEmbedding(
            model=model_identifier,
            temperature=temperature,
            deployment_name=secrets.get("azure_embedding_deployment_name"),
            api_key=secrets.get("azure_api_key"),
            azure_endpoint=secrets.get("azure_api_endpoint_url"),
            api_version=secrets.get("azure_api_version")
        )
        return azure_embedding_model
            
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

    def load_model(self, model_category: str, model_prefix: str) -> AzureOpenAI:
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
           model = self._instantiate_azure_llm_model(self._secrets, model_prefix) 
        elif model_category == "embedding":
            model = self._instantiate_azure_embedding_model(self._secrets, model_prefix)
        else:
            raise NotImplementedError(f"Model Category {model_category} not implemented")

        assert model is not None, "Model not loaded"
        return model