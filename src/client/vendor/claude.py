from loguru import logger
from typing import Optional
from utils import methods

from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from anthropic.lib.vertex._client import AnthropicVertex



class Claude(CustomLLM):
    context_window: int = 4096
    num_output: int = 256
    model_name: str = "custom"
    client: Optional[AnthropicVertex] = None

    def __init__(self):
        super().__init__()
        self.client = None  # Initialize as None, will be created when needed

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _ensure_client(self):
        if self.client is None:
            self.client = AnthropicVertex(region="europe-west1", project_id="website-254017")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self._ensure_client()
        message = self.client.messages.create(
            max_tokens=4096,
            messages=[
                {
                "role": "user",
                "content": prompt,
                }
            ],
            model="claude-3-5-sonnet@20240620",
        )
        return CompletionResponse(text=message.content[0].text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        self._ensure_client()
        response = ""
        with self.client.messages.stream(
            max_tokens=4096,
            messages=[
                {
                "role": "user",
                "content": prompt,
                }
            ],
            model="claude-3-5-sonnet@20240620",
        ) as stream:
            for text in stream.text_stream:
                response += text
                yield CompletionResponse(text=response, delta=text)

class ClaudeClient:
    def __init__(self) -> None:
        """
        Initialize an instance of AzureClient.

        Args:
            None

        Returns:
            None
        """
        self._secrets = "lll"
        # Internal State
        self._self_hosted = False

    def _instantiate_openai_llm_model(self,
        secrets: dict, model_identifier: Optional[str] = "llama3-70b-8192"
    ) -> Claude:
        """
        Instantiate an AzureOpenAI model for Language Model (LLM) using the provided secrets and model identifier.

        Args:
            secrets (dict): The dictionary containing the necessary credentials and configuration.
            model_identifier (str, optional): The identifier of the model to be used. Defaults to "gpt-35-turbo".

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model for Language Model (LLM).
        """
        llm_model = Claude()
        return llm_model

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

    def load_model(self, model_category: str = "", model_prefix: str="") -> Claude:
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