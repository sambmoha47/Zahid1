from llama_index.llms.groq import Groq

from loguru import logger
from typing import Optional
from utils import methods


class GroqClient:
    def __init__(self) -> None:
        """
        Initialize an instance of AzureClient.

        Args:
            None

        Returns:
            None
        """
        self._secrets = methods.extract_grok_secrets()
        # Internal State
        self._self_hosted = False

    def _instantiate_openai_llm_model(self,
        secrets: dict, model_identifier: Optional[str] = "llama3-70b-8192"
    ) -> Groq:
        """
        Instantiate an AzureOpenAI model for Language Model (LLM) using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "gpt-35-turbo".

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model for Language Model (LLM).
        """
        grok_llm_model = Groq(model=model_identifier, api_key=secrets["grok_api_key"])
        return grok_llm_model

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

    def load_model(self, model_category: str, model_prefix: str) -> Groq:
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
           model = self._instantiate_openai_llm_model(self._secrets, model_prefix) 
        else:
            raise NotImplementedError(f"Model Category {model_category} not implemented")

        assert model is not None, "Model not loaded"
        return model