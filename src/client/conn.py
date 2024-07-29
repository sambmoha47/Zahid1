from typing import Optional, Union, Callable
from pathlib import PosixPath
from node import Config

from .vendor import AzureClient, HuggingFaceClient, VertexClient, OpenAIClient, PineconeClient, GroqClient, ClaudeClient

__KNOWN_VENDORS = ["azure", "huggingface", "openai", "vertex", "grok", 'claude']


class ClientConnector:
    def __init__(
        self,
        embed_client: Callable,
        embed_model: str,
        llm_client: Callable,
        llm: str,
        pinecone_client: Callable
    ) -> None:
        """
        Initialize the ClientConnector class.

        Args:
            embed_client (Callable): The embedding client.
            embed_model (str): The embedding model.
            llm_client (Callable): The LLM client.
            llm (str): The LLM model.
        """
        self._embed_client = embed_client
        self._embed_model = embed_model
        self._llm_client = llm_client
        self._llm = llm
        self._pinecone_client = pinecone_client

    def load_embed_model(self):
        """
        Load the embedding model.

        Returns:
            The loaded embedding model.
        """
        embed_model = self._embed_client.load_model(model_category="embedding", model_prefix=self._embed_model)
        # FIXME: make harder check instead of None
        assert embed_model is not None, "Embedding model is missing."
        return embed_model

    def load_llm(self):
        """
        Load the LLM model.

        Returns:
            The loaded LLM model.
        """
        llm = self._llm_client.load_model(model_category="llm", model_prefix=self._llm)
        # FIXME: make harder check instead of None
        assert llm is not None, "LLM model is missing."
        return llm

    def load_pinecone_client(self):
        """
        Load the Pinecone client.

        Returns:
            The loaded Pinecone client.
        """
        return self._pinecone_client 


def instantiate_client_connector(
    conf: Config, secrets_dir: Union[bool, PosixPath] = None
) -> ClientConnector:
    """
    Instantiate the ClientConnector class.

    Args:
        conf (Config): The configuration object.
        secrets_dir (Union[bool, PosixPath], optional): The secrets directory. Defaults to None.

    Returns:
        The instantiated ClientConnector object.
    """
    _conf = conf.models
    assert _conf is not None, "Model configuration is missing."
    if secrets_dir is not None:
        assert secrets_dir.exists(), f"Secrets directory {secrets_dir} does not exist."
    else:
        # FIXME update vendors to use new secrets dir
        pass

    embed_conf = _conf.get("embed", None)
    llm_conf = _conf.get("llm", None)
    assert embed_conf is not None, "Embedding configuration is missing."
    assert llm_conf is not None, "LLM configuration is missing."

    embed_client_conf = embed_conf.client
    embed_model = embed_conf.prefix
    llm_client_conf = llm_conf.client
    llm = llm_conf.prefix

    assert embed_client_conf is not None, "Embedding client is missing."
    assert llm_client_conf is not None, "LLM client is missing."
    assert (
        embed_client_conf in __KNOWN_VENDORS
    ), f"Embedding client {embed_client_conf} is not supported."
    assert llm_client_conf in __KNOWN_VENDORS, f"LLM client {llm_client_conf} is not supported."
    assert embed_model is not None, "Embedding model is missing."
    assert llm is not None, "LLM model is missing."

    embed_client, llm_client = _resolve_clients(embed_client_conf, llm_client_conf)
    pinecone_client = PineconeClient()
    pinecone_client.connect()

    return ClientConnector(
        embed_client=embed_client,
        embed_model=embed_model,
        llm_client=llm_client,
        llm=llm,
        pinecone_client=pinecone_client,
    )


def _resolve_clients(embed_client_conf: str, llm_client_conf: str):
    """
    Resolve the embedding and LLM clients.

    Args:
        embed_client_conf (str): The embedding client configuration.
        llm_client_conf (str): The LLM client configuration.

    Returns:
        The embedding and LLM clients.
    """
    embed_client = None
    if embed_client_conf == "huggingface":
        embed_client = HuggingFaceClient()
    elif embed_client_conf == "vertex":
        embed_client = VertexClient()
    elif embed_client_conf == "azure":
        embed_client = AzureClient()
    elif embed_client_conf == "openai":
        embed_client = OpenAIClient()
    else:
        raise ValueError(f"Embedding client {embed_client} is not supported.")
    assert embed_client is not None, "Embedding client is missing."
    embed_client.connect()

    llm_client = None
    if llm_client_conf == "huggingface":
        llm_client = HuggingFaceClient()
    elif llm_client_conf == "vertex":
        llm_client = VertexClient()
    elif llm_client_conf == "azure":
        llm_client = AzureClient()
    elif llm_client_conf == "openai":
        llm_client = OpenAIClient()
    elif llm_client_conf == "grok":
        llm_client = GroqClient()
    elif llm_client_conf == 'claude':
        llm_client = ClaudeClient()
    else:
        raise ValueError(f"LLM client {llm_client} is not supported.")
    assert llm_client is not None, "LLM client is missing."
    llm_client.connect()
    return embed_client, llm_client

# from typing import Optional, Union, Callable
# from pathlib import PosixPath
# from node import Config

# from .vendor import AzureClient, HuggingFaceClient, VertexClient, OpenAIClient, PineconeClient
# from tensorflow_model_optimization.sparsity import keras as sparsity
# import tensorflow as tf
# from cachetools import cached, LRUCache

# __KNOWN_VENDORS = ["azure", "huggingface", "openai", "vertex"]

# # Cache setup
# cache = LRUCache(maxsize=1000)

# def convert_to_keras_model(model):
#     # If model is already a Sequential or functional model, return it directly
#     if isinstance(model, (tf.keras.models.Sequential, tf.keras.Model)):
#         return model
#     # If the model is a list of layers, convert it to a Sequential model
#     elif isinstance(model, list) and all(isinstance(layer, tf.keras.layers.Layer) for layer in model):
#         return tf.keras.models.Sequential(model)
#     # If model type is not supported, raise an error
#     else:
#         raise ValueError(f"Cannot convert model of type {type(model)} to a supported Keras model type.")

# def prune_and_quantize_model(model):
#     # Apply pruning
#     pruning_params = {
#         'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
#                                                      final_sparsity=0.90,
#                                                      begin_step=2000,
#                                                      end_step=10000)
#     }

#     # Convert model to a supported type
#     model = convert_to_keras_model(model)
    
#     pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

#     # Apply quantization
#     converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     quantized_model = converter.convert()

#     return quantized_model

# @cached(cache)
# def get_embedding_cached(embedding_model, query):
#     return embedding_model(query)

# class ClientConnector:
#     def __init__(
#         self,
#         embed_client: Callable,
#         embed_model: str,
#         llm_client: Callable,
#         llm: str,
#         pinecone_client: Callable
#     ) -> None:
#         """
#         Initialize the ClientConnector class.

#         Args:
#             embed_client (Callable): The embedding client.
#             embed_model (str): The embedding model.
#             llm_client (Callable): The LLM client.
#             llm (str): The LLM model.
#         """
#         self._embed_client = embed_client
#         self._embed_model = embed_model
#         self._llm_client = llm_client
#         self._llm = llm
#         self._pinecone_client = pinecone_client

#     def load_embed_model(self):
#         """
#         Load the pruned and quantized embedding model.

#         Returns:
#             The loaded embedding model.
#         """
#         embed_model = self._embed_client.load_model(model_category="embedding", model_prefix=self._embed_model)
        
#         embed_model = prune_and_quantize_model(embed_model)
#         assert embed_model is not None, "Embedding model is missing."
#         return embed_model

#     def load_llm(self):
#         """
#         Load the LLM model.

#         Returns:
#             The loaded LLM model.
#         """
#         llm = self._llm_client.load_model(model_category="llm", model_prefix=self._llm)
#         assert llm is not None, "LLM model is missing."
#         return llm

#     def load_pinecone_client(self):
#         """
#         Load the Pinecone client.

#         Returns:
#             The loaded Pinecone client.
#         """
#         return self._pinecone_client


# def instantiate_client_connector(
#     conf: Config, secrets_dir: Union[bool, PosixPath] = None
# ) -> ClientConnector:
#     """
#     Instantiate the ClientConnector class.

#     Args:
#         conf (Config): The configuration object.
#         secrets_dir (Union[bool, PosixPath], optional): The secrets directory. Defaults to None.

#     Returns:
#         The instantiated ClientConnector object.
#     """
#     _conf = conf.models
#     assert _conf is not None, "Model configuration is missing."
#     if secrets_dir is not None:
#         assert secrets_dir.exists(), f"Secrets directory {secrets_dir} does not exist."
#     else:
#         # FIXME update vendors to use new secrets dir
#         pass

#     embed_conf = _conf.get("embed", None)
#     llm_conf = _conf.get("llm", None)
#     assert embed_conf is not None, "Embedding configuration is missing."
#     assert llm_conf is not None, "LLM configuration is missing."

#     embed_client_conf = embed_conf.client
#     embed_model = embed_conf.prefix
#     llm_client_conf = llm_conf.client
#     llm = llm_conf.prefix

#     assert embed_client_conf is not None, "Embedding client is missing."
#     assert llm_client_conf is not None, "LLM client is missing."
#     assert (
#         embed_client_conf in __KNOWN_VENDORS
#     ), f"Embedding client {embed_client_conf} is not supported."
#     assert llm_client_conf in __KNOWN_VENDORS, f"LLM client {llm_client_conf} is not supported."
#     assert embed_model is not None, "Embedding model is missing."
#     assert llm is not None, "LLM model is missing."

#     embed_client, llm_client = _resolve_clients(embed_client_conf, llm_client_conf)
#     pinecone_client = PineconeClient()
#     pinecone_client.connect()

#     return ClientConnector(
#         embed_client=embed_client,
#         embed_model=embed_model,
#         llm_client=llm_client,
#         llm=llm,
#         pinecone_client=pinecone_client,
#     )


# def _resolve_clients(embed_client_conf: str, llm_client_conf: str):
#     """
#     Resolve the embedding and LLM clients.

#     Args:
#         embed_client_conf (str): The embedding client configuration.
#         llm_client_conf (str): The LLM client configuration.

#     Returns:
#         The embedding and LLM clients.
#     """
#     embed_client = None
#     if embed_client_conf == "huggingface":
#         embed_client = HuggingFaceClient()
#     elif embed_client_conf == "vertex":
#         embed_client = VertexClient()
#     elif embed_client_conf == "azure":
#         embed_client = AzureClient()
#     elif embed_client_conf == "openai":
#         embed_client = OpenAIClient()
#     else:
#         raise ValueError(f"Embedding client {embed_client} is not supported.")
#     assert embed_client is not None, "Embedding client is missing."
#     embed_client.connect()

#     llm_client = None
#     if llm_client_conf == "huggingface":
#         llm_client = HuggingFaceClient()
#     elif llm_client_conf == "vertex":
#         llm_client = VertexClient()
#     elif llm_client_conf == "azure":
#         llm_client = AzureClient()
#     elif llm_client_conf == "openai":
#         llm_client = OpenAIClient()
#     else:
#         raise ValueError(f"LLM client {llm_client} is not supported.")
#     assert llm_client is not None, "LLM client is missing."
#     llm_client.connect()
#     return embed_client, llm_client
