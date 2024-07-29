import numpy as np
from typing import Optional, Dict, List
from pathlib import PosixPath, Path
from loguru import logger
from node import Config
from node import read_configuration
from client import instantiate_client_connector, ClientConnector
from parser import transform_documents, load_documents
from template import GENERIC_PROMPT_TEMPLATE, CONTEXT_AWARE_PROMPT_TEMPLATE, CONTEXT_AND_LANGUAGE_AWARE_TEMPLATE, DOC_TEMPLATE
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
import pickle
import os

class Pipeline:
    """
    Agent class represents an intelligent agent that interacts with the system.
    """

    def __init__(self, conf: Config) -> None:
        """
        Initializes the Agent object.

        Args:
            conf (Config): The configuration object.

        Returns:
            None
        """
        self._conf: Config = conf

        # Internal State
        self._index_conf = None
        self._client_conf = None
        self._parser_conf = None

        # Connectors and Models
        self._client = None
        self._embed_model = None
        self._llm = None

        # Indexes
        self._indexes = None

        # Vector db client
        self._vector_db_client = None

        self.fail_embeds = None
        self.general_embeds = None
    
    @classmethod
    def from_conf(cls, conf_path: PosixPath) -> 'Pipeline':
        """
        Creates an Agent object from a configuration.

        Args:
            conf (Config): The configuration object.

        Returns:
            Agent: The created Agent object.
        """
        configuration: Config = read_configuration(conf_path)
        instance = cls(configuration)
        instance._index_conf = configuration.index
        instance._client_conf = configuration.client
        instance._parser_conf = configuration.parser
        assert instance._index_conf is not None, "Index configuration is missing."
        assert instance._client_conf is not None, "Client configuration is missing."
        assert instance._parser_conf is not None, "Parser configuration is missing."
        return instance

    def connect_client(self, secrets_directory: PosixPath) -> None:
        """
        Connects the client to the server.

        Args:
            secrets_file_name (str): The name of the secrets file.

        Returns:
            None
        """
        assert secrets_directory.exists(), "Secrets directory does not exist."
        self._client: ClientConnector = instantiate_client_connector(self._client_conf, secrets_directory) 

    def prepare_settings(self):
        embed_model = self._client.load_embed_model()
        llm = self._client.load_llm()
        self._llm = llm
        self._embed_model = embed_model

        self._vector_db_client = self._client.load_pinecone_client()

        Settings.llm = self._llm
        Settings.embed_model = self._embed_model
        logger.info(f"[Settings]\n{Settings}")

    def prepare_embeddings(self, parser_type: Optional[str] = "base", update_on_change: Optional[bool] = False):
        if update_on_change:
            raise NotImplementedError("Update on change is not yet implemented.")

        if self._index_conf.index_type == "multiple":
            index_names = self._index_conf.folder_indexes
            index_complete_path = Path("data/vector")
            folder_complete_path = Path("data/source")
            use_existing_index = self._index_conf.load_existing_index_under_prefix
            indexes = self._prepare_multiple_index(
                index_names=index_names,
                index_complete_path=index_complete_path,
                folder_complete_path=folder_complete_path,
                use_existing_index=use_existing_index,
                parser_type=parser_type,
            )
        elif self._index_conf.index_type == "single":
            raise NotImplementedError("Single index is not yet implemented.")
        else:
            raise ValueError(f"Invalid index type: {self._index_conf.index_type}")
        self._indexes = indexes

    def spawn_query_engine(self, index_identifier: Optional[str] = "commercial_index"):
        assert self._indexes is not None, "Indexes are not yet prepared."

        # FIXME: should be taken from the configuration
        index = None
        if index_identifier == "all":
            raise NotImplementedError("Querying all indexes is not yet implemented.")
        elif index_identifier == "commercial-index":
            index = self._indexes.get("commercial-index", None)
        elif index_identifier == "zahid-index":
            index = self._indexes.get("zahid-index", None)
        elif index_identifier == "mostostal-index":
            index = self._indexes.get("mostostal-index", None)
        else:
            raise ValueError(f"Invalid index identifier: {index_identifier}")
        assert index is not None, "Index is missing."
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="simple_summarize", streaming=True)
        # from IPython.display import Markdown, display
        # def display_prompt_dict(prompts_dict):
        #     for k, p in prompts_dict.items():
        #         text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        #         print(text_md)
        #         # display(Markdown(text_md))
        #         print(p.get_template())
        #         # display(Markdown("<br><br>"))
        # print("QQQQQQQQQ:\n", display_prompt_dict(query_engine.get_prompts()))
        
        # new_summary_tmpl_str = (
        #     "Context information is below.\n"
        #     "---------------------\n"
        #     "{context_str}\n"
        #     "---------------------\n"
        #     "Given the context information and not prior knowledge, "
        #     "answer the query. Answer the query as detailed as possible with all the relevant information about all the points included in the answer.\n"
        #     "Query: {query_str}\n"
        #     "Answer: "
        # )
        # new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
        # query_engine.update_prompts(
        #     {"text_qa_template": new_summary_tmpl}
        # )
        # print("QQQQQQQQQ:\n", display_prompt_dict(query_engine._response_synthesizer.get_prompts()))
        self._update_engine_prompt(query_engine, prompt="doc", update_field='response_synthesizer:text_qa_template')
        # print("QQQQQQQQQ:\n", display_prompt_dict(query_engine.get_prompts()))
        return query_engine
    
    def spawn_chat_engine(self, index_identifier: Optional[str] = "commercial_index"):
        assert self._indexes is not None, "Indexes are not yet prepared."

        # FIXME: should be taken from the configuration
        index = None
        if index_identifier == "all":
            raise NotImplementedError("Querying all indexes is not yet implemented.")
        elif index_identifier == "commercial-index":
            index = self._indexes.get("commercial-index", None)
        elif index_identifier == "zahid-index":
            index = self._indexes.get("zahid-index", None)
        elif index_identifier == "mostostal-index":
            index = self._indexes.get("mostostal-index", None)
        else:
            raise ValueError(f"Invalid index identifier: {index_identifier}")
        assert index is not None, "Index is missing."
        memory = ChatMemoryBuffer.from_defaults(token_limit=8_000)
        chat_engine = index.as_chat_engine(
            chat_mode="react",
            verbose=True,
            memory=memory,
            system_prompt = (
                """You are a virtual assistant designated for the Zahid Group. Your primary function is to field inquiries
                pertaining to business operations, human resources, company policies, and other aspects relevant to the organization.
                Your responses should accurately address acronyms, abbreviations, and specialized jargon to maintain clear communication.
                Be vigilant in understanding the context and details of each inquiry.
                Please be aware that some questions may contain acronyms, abbreviations, or specialized terminology.
                If a query is unclear or lacks information, do not hesitate to request additional details from the user.
                Adhere to the following two critical guidelines:\n
                1. Focus exclusively on topics directly related to the Zahid Group and source documents provided.\n
                2. Respond to inquiries in the same language in which they are asked to ensure effective communication."""
            ), 

        )
        return chat_engine

    def load_embeddings(self):
        general_question_templates = [
            "What documents do you have?",
            "What service can you provide?",
            "Can you list the documents currently available?",
            "What types of service do you offer?",
            "What documents are stored in this system?",
            "Can you describe the services you provide?",
            "What are the available documents in your archive?",
            "What specific services can this system offer?",
            "What document collections do you have?",
            "What can you provide in terms of service?"
            "What files do you have"
            "List all the files"
            "Please list all the files you have",
            "What is your file database",
        ]

        if self.general_embeds is None:
            if os.path.exists('general_embeds.pkl'):
                with open('general_embeds.pkl', 'rb') as f:
                    self.general_embeds = pickle.load(f)
            else:
                print("running full")
                self.general_embeds = self._embed_model.get_agg_embedding_from_queries(general_question_templates)
                with open('general_embeds.pkl', 'wb') as f:
                    pickle.dump(self.general_embeds, f)

        fail_ans_templates = [
            "the provided context does not include specific information",
            "not provided in the available documents",
            "context does not include specific information",
            "the context does not include specific information about",
        ]

        if self.fail_embeds is None:
            # self.fail_embeds = self._embed_model.get_agg_embedding_from_queries(fail_ans_templates)
            if os.path.exists('fail_embed.pkl'):
                # Load from pickle file if it exists
                with open("fail_embed.pkl", 'rb') as f:
                    self.fail_embeds = pickle.load(f)
            else:
                # Generate embeddings if pickle file doesn't exist
                print("running full")
                self.fail_embeds = self._embed_model.get_agg_embedding_from_queries(fail_ans_templates)
                
                # Save to pickle file
                with open("fail_embed.pkl", 'wb') as f:
                    pickle.dump(self.fail_embeds, f)
        

    def check_if_user_asks_about_general_info(self, message: str) -> bool:
        """
        Checks if the user asks about general information.

        Args:
            message (str): The message from the user.

        Returns:
            bool: True if the user asks about general information, False otherwise.
        """



        # if self.general_embeds is None:
        #     self.general_embeds = self._embed_model.get_agg_embedding_from_queries(general_question_templates)
        import time
        strt = time.time()
        user_embeds = self._embed_model.get_agg_embedding_from_queries([message])
        sim = self._embed_model.similarity(self.general_embeds, user_embeds)
        end = time.time()
        print("TIME: ", end-strt)
        logger.warning(f"[SIMILARITY] User question to general similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User question to general similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User question to general similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User question to general similarity score: {sim}")
        return sim > 0.71

    def check_if_retrieval_failed(self, response: str) -> bool:
        """
        Checks if the answer is a failure.

        Args:
            response (str): The response from the system.

        Returns:
            bool: True if the response is a failure, False otherwise.
        """
        # if self.fail_embeds is None:
        #     self.fail_embeds = self._embed_model.get_agg_embedding_from_queries(fail_ans_templates)
        import time
        strt = time.time()
        user_embeds = self._embed_model.get_agg_embedding_from_queries([response])
        sim = self._embed_model.similarity(self.fail_embeds, user_embeds)
        end = time.time()
        print("TIME: ", end-strt)
        logger.warning(f"[SIMILARITY] User response to failure similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User response to failure similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User response to failure similarity score: {sim}")
        # logger.warning(f"[SIMILARITY] User response to failure similarity score: {sim}")
        return sim > 0.7

    # ---- ---- ---- ---- ---- <
    # > Private Methods
    # ---- ---- ---- ---- ---- <
        
    # FIXME: parser_type should be a part of the configuration
    def _load_data(self, *, source_path, parser_type: str) -> Dict[str, List[Document]]:
        logger.warning(f"[LOADING DATA] Loading data using parser type: {parser_type}")
        documents: dict[str, list[Document]]= load_documents(source_path=source_path, parser_type=parser_type)
        return documents

    def _prepare_multiple_index(self, index_names: list[str], index_complete_path: PosixPath, folder_complete_path: PosixPath, use_existing_index: Optional[bool] = True, parser_type: Optional[str] = "base") -> Dict[str, VectorStoreIndex]:
        """
        Prepares multiple indexes, one for each key in the documents dictionary.
        """
        indexes = {}
        for index in index_names:
            folder_name = index.get("folder", None)
            index_name = index.get("index_name", None)
            assert folder_name is not None, "Folder name is missing."
            assert index_name is not None, "Index name is missing." 

            index_exists_in_pinecone = self._vector_db_client.index_exists(index_name=index_name)
            if use_existing_index and index_exists_in_pinecone:
                logger.warning("[INITIALIZATION] Utilizing the existing indexes.")
                vector_store = self._vector_db_client.get_existing_vstore(index_name)
                index = VectorStoreIndex.from_vector_store(vector_store)
                logger.warning("[INITIALIZATION COMPLETE] Initialization of existing indexes complete.")
            else:
                logger.warning("[CREATION] Commencing new indexes creation.")

                # FIXME: This should be taken from the configuration
                delete_index = False
                if delete_index:
                    self._vector_db_client.delete_index(index_name=index_name)

                vector_store = self._vector_db_client.create_new_vstore(index_name=index_name, dim=768)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                folder_complete_path_per_key = folder_complete_path / folder_name
                docs = self._load_data(source_path=folder_complete_path_per_key, parser_type=parser_type)
                nodes = transform_documents(docs, llm=self._llm, conf=self._parser_conf)
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    use_async=True,
                    show_progress=True,
                )
                logger.warning("[CREATION COMPLETE] New indexes have been created.")
            indexes[index_name] = index
            logger.warning(f"[INDEX INFO] WORKING WITH INDEXES: \n{indexes.keys()}")

        assert indexes, "No indexes were created."
        return indexes

    def _prepare_single_index(self, documents: List[Document], storage_context: StorageContext) -> VectorStoreIndex:
        raise NotImplementedError("Single index is not yet implemented.")

    def _update_engine_prompt(self, engine, prompt: Optional[str] ="generic", update_field: Optional[str]='response_synthesizer:summary_template') -> None:
        refined_prompt_template = None
        if prompt == "generic":
            refined_prompt_template = PromptTemplate(GENERIC_PROMPT_TEMPLATE)
        elif prompt == "context_aware":
            refined_prompt_template = PromptTemplate(CONTEXT_AWARE_PROMPT_TEMPLATE)
        elif prompt == "context_and_language_aware":
            refined_prompt_template = PromptTemplate(CONTEXT_AND_LANGUAGE_AWARE_TEMPLATE)
        elif prompt == "doc":
            refined_prompt_template = PromptTemplate(DOC_TEMPLATE)
        else:
            raise ValueError(f"Invalid prompt: {prompt}")
        assert refined_prompt_template is not None, "Prompt template is missing."

        engine.update_prompts(
            {update_field: refined_prompt_template},
        )

    def _prepare_qdrant(self) -> None:
        pass

# import faiss
# import numpy as np
# from typing import Optional, Dict, List
# from pathlib import PosixPath, Path
# from loguru import logger
# from node import Config
# from node import read_configuration
# from client import instantiate_client_connector, ClientConnector
# from parser import transform_documents, load_documents
# from template import GENERIC_PROMPT_TEMPLATE, CONTEXT_AWARE_PROMPT_TEMPLATE, CONTEXT_AND_LANGUAGE_AWARE_TEMPLATE, \
#     DOC_TEMPLATE

# from llama_index.core import Document
# from llama_index.core import Settings
# from llama_index.core import VectorStoreIndex
# from llama_index.core.storage import StorageContext
# from llama_index.core import load_index_from_storage
# from llama_index.core import PromptTemplate
# from llama_index.core.memory import ChatMemoryBuffer

# from cachetools import cached, LRUCache
# import asyncio

# # Cache setup
# cache = LRUCache(maxsize=1000)


# async def async_get_embedding(embedding_model, query):
#     return embedding_model(query)


# async def process_embeddings(embedding_model, queries):
#     tasks = [async_get_embedding(embedding_model, query) for query in queries]
#     results = await asyncio.gather(*tasks)
#     return results


# @cached(cache)
# def get_embedding_cached(embedding_model, query):
#     return embedding_model(query)


# class Pipeline:
#     """
#     Agent class represents an intelligent agent that interacts with the system.
#     """

#     def __init__(self, conf: Config) -> None:
#         """
#         Initializes the Agent object.

#         Args:
#             conf (Config): The configuration object.

#         Returns:
#             None
#         """
#         self._conf: Config = conf

#         # Internal State
#         self._index_conf = None
#         self._client_conf = None
#         self._parser_conf = None

#         # Connectors and Models
#         self._client = None
#         self._embed_model = None
#         self._llm = None

#         # Indexes
#         self._indexes = None

#         # Vector db client
#         self._vector_db_client = None

#     @classmethod
#     def from_conf(cls, conf_path: PosixPath) -> 'Pipeline':
#         """
#         Creates an Agent object from a configuration.

#         Args:
#             conf (Config): The configuration object.

#         Returns:
#             Agent: The created Agent object.
#         """
#         configuration: Config = read_configuration(conf_path)
#         instance = cls(configuration)
#         instance._index_conf = configuration.index
#         instance._client_conf = configuration.client
#         instance._parser_conf = configuration.parser
#         assert instance._index_conf is not None, "Index configuration is missing."
#         assert instance._client_conf is not None, "Client configuration is missing."
#         assert instance._parser_conf is not None, "Parser configuration is missing."
#         return instance

#     def connect_client(self, secrets_directory: PosixPath) -> None:
#         """
#         Connects the client to the server.

#         Args:
#             secrets_file_name (str): The name of the secrets file.

#         Returns:
#             None
#         """
#         assert secrets_directory.exists(), "Secrets directory does not exist."
#         self._client: ClientConnector = instantiate_client_connector(self._client_conf, secrets_directory)

#     def prepare_settings(self):
#         embed_model = self._client.load_embed_model()
#         llm = self._client.load_llm()
#         self._llm = llm
#         self._embed_model = embed_model

#         self._vector_db_client = self._client.load_pinecone_client()

#         Settings.llm = self._llm
#         Settings.embed_model = self._embed_model
#         logger.info(f"[Settings]\n{Settings}")

#     async def prepare_embeddings_async(self, parser_type: Optional[str] = "base",
#                                        update_on_change: Optional[bool] = False):
#         if update_on_change:
#             raise NotImplementedError("Update on change is not yet implemented.")

#         if self._index_conf.index_type == "multiple":
#             index_names = self._index_conf.folder_indexes
#             index_complete_path = Path("data/vector")
#             folder_complete_path = Path("data/source")
#             use_existing_index = self._index_conf.load_existing_index_under_prefix
#             indexes = await self._prepare_multiple_index_async(
#                 index_names=index_names,
#                 index_complete_path=index_complete_path,
#                 folder_complete_path=folder_complete_path,
#                 use_existing_index=use_existing_index,
#                 parser_type=parser_type,
#             )
#         elif self._index_conf.index_type == "single":
#             raise NotImplementedError("Single index is not yet implemented.")
#         else:
#             raise ValueError(f"Invalid index type: {self._index_conf.index_type}")

#         self._indexes = indexes

#     async def _prepare_multiple_index_async(self, index_names: list[str], index_complete_path: PosixPath,
#                                             folder_complete_path: PosixPath, use_existing_index: Optional[bool] = True,
#                                             parser_type: Optional[str] = "base") -> Dict[str, VectorStoreIndex]:
#         indexes = {}
#         for index in index_names:
#             folder_name = index.get("folder", None)
#             index_name = index.get("index_name", None)
#             assert folder_name is not None, "Folder name is missing."
#             assert index_name is not None, "Index name is missing."

#             index_exists_in_pinecone = self._vector_db_client.index_exists(index_name=index_name)
#             if use_existing_index and index_exists_in_pinecone:
#                 logger.warning("[INITIALIZATION] Utilizing the existing indexes.")
#                 vector_store = self._vector_db_client.get_existing_vstore(index_name)
#                 index = VectorStoreIndex.from_vector_store(vector_store)
#                 logger.warning("[INITIALIZATION COMPLETE] Initialization of existing indexes complete.")
#             else:
#                 logger.warning("[CREATION] Commencing new indexes creation.")

#                 delete_index = False
#                 if delete_index:
#                     self._vector_db_client.delete_index(index_name=index_name)

#                 vector_store = self._vector_db_client.create_new_vstore(index_name=index_name, dim=768)
#                 storage_context = StorageContext.from_defaults(vector_store=vector_store)
#                 folder_complete_path_per_key = folder_complete_path / folder_name
#                 documents = await self._load_data_async(source_path=folder_complete_path_per_key,
#                                                         parser_type=parser_type)
#                 queries = [doc.text for doc in documents]

#                 # Batch processing for embeddings
#                 embeddings = []
#                 batch_size = 32  # Adjust based on system capacity
#                 for i in range(0, len(queries), batch_size):
#                     batch = queries[i:i + batch_size]
#                     batch_embeddings = await process_embeddings(self._embed_model, batch)
#                     embeddings.extend(batch_embeddings)

#                 faiss_index = create_faiss_index(np.array(embeddings), dimension=768)
#                 index = VectorStoreIndex(
#                     nodes=faiss_index,
#                     storage_context=storage_context,
#                     use_async=True,
#                     show_progress=True,
#                 )
#                 logger.warning("[CREATION COMPLETE] New indexes have been created.")
#             indexes[index_name] = index
#             logger.warning(f"[INDEX INFO] WORKING WITH INDEXES: \n{indexes.keys()}")

#         assert indexes, "No indexes were created."
#         return indexes

#     def spawn_query_engine(self, index_identifier: Optional[str] = "commercial_index"):
#         assert self._indexes is not None, "Indexes are not yet prepared."

#         # FIXME: should be taken from the configuration
#         index = None
#         if index_identifier == "all":
#             raise NotImplementedError("Querying all indexes is not yet implemented.")
#         elif index_identifier == "commercial-index":
#             index = self._indexes.get("commercial-index", None)
#         elif index_identifier == "zahid-index":
#             index = self._indexes.get("zahid-index", None)
#         elif index_identifier == "mostostal-index":
#             index = self._indexes.get("mostostal-index", None)
#         else:
#             raise ValueError(f"Invalid index identifier: {index_identifier}")
#         assert index is not None, "Index is missing."
#         query_engine = index.as_query_engine(similarity_top_k=9, response_mode="tree_summarize")
#         self._update_engine_prompt(query_engine, prompt="doc")
#         return query_engine

#     def spawn_chat_engine(self, index_identifier: Optional[str] = "commercial_index"):
#         assert self._indexes is not None, "Indexes are not yet prepared."

#         # FIXME: should be taken from the configuration
#         index = None
#         if index_identifier == "all":
#             raise NotImplementedError("Querying all indexes is not yet implemented.")
#         elif index_identifier == "commercial-index":
#             index = self._indexes.get("commercial-index", None)
#         elif index_identifier == "zahid-index":
#             index = self._indexes.get("zahid-index", None)
#         elif index_identifier == "mostostal-index":
#             index = self._indexes.get("mostostal-index", None)
#         else:
#             raise ValueError(f"Invalid index identifier: {index_identifier}")
#         assert index is not None, "Index is missing."
#         memory = ChatMemoryBuffer.from_defaults(token_limit=8_000)
#         chat_engine = index.as_chat_engine(
#             chat_mode="react",
#             verbose=True,
#             memory=memory,
#             system_prompt=(
#                 "You are a virtual assistant designated for the Zahid Group. Your primary function is to field inquiries "
#                 "pertaining to business operations, human resources, company policies, and other aspects relevant to the organization. "
#                 "Your responses should accurately address acronyms, abbreviations, and specialized jargon to maintain clear communication. "
#                 "Be vigilant in understanding the context and details of each inquiry. "
#                 "Please be aware that some questions may contain acronyms, abbreviations, or specialized terminology. "
#                 "If a query is unclear or lacks information, do not hesitate to request additional details from the user. "
#                 "Adhere to the following two critical guidelines:\n"
#                 "1. Focus exclusively on topics directly related to the Zahid Group and source documents provided.\n"
#                 "2. Respond to inquiries in the same language in which they are asked to ensure effective communication."
#             ),

#         )
#         return chat_engine

#     def check_if_user_asks_about_general_info(self, message: str) -> bool:
#         """
#         Checks if the user asks about general information.

#         Args:
#             message (str): The message from the user.

#         Returns:
#             bool: True if the user asks about general information, False otherwise.
#         """

#         general_question_templates = [
#             "What documents do you have?",
#             "What service can you provide?",
#             "Can you list the documents currently available?",
#             "What types of service do you offer?",
#             "What documents are stored in this system?",
#             "Can you describe the services you provide?",
#             "What are the available documents in your archive?",
#             "What specific services can this system offer?",
#             "What document collections do you have?",
#             "What can you provide in terms of service?"
#             "What files do you have"
#             "List all the files"
#             "Please list all the files you have",
#             "What is your file database",
#         ]

#         general_embeds = self._embed_model.get_agg_embedding_from_queries(general_question_templates)
#         user_embeds = self._embed_model.get_agg_embedding_from_queries([message])
#         sim = self._embed_model.similarity(general_embeds, user_embeds)
#         logger.warning(f"[SIMILARITY] User question to general similarity score: {sim}")
#         return sim > 0.71

#     def check_if_retrieval_failed(self, response: str) -> bool:
#         """
#         Checks if the answer is a failure.

#         Args:
#             response (str): The response from the system.

#         Returns:
#             bool: True if the response is a failure, False otherwise.
#         """
#         fail_ans_templates = [
#             "the provided context does not include specific information",
#             "not provided in the available documents",
#             "context does not include specific information",
#             "the context does not include specific information about",
#         ]

#         fail_embeds = self._embed_model.get_agg_embedding_from_queries(fail_ans_templates)
#         user_embeds = self._embed_model.get_agg_embedding_from_queries([response])
#         sim = self._embed_model.similarity(fail_embeds, user_embeds)
#         logger.warning(f"[SIMILARITY] User response to failure similarity score: {sim}")
#         return sim > 0.7

#     # ---- ---- ---- ---- ---- <
#     # > Private Methods
#     # ---- ---- ---- ---- ---- <

#     async def _load_data_async(self, *, source_path, parser_type: str) -> Dict[str, List[Document]]:
#         logger.warning(f"[LOADING DATA] Loading data using parser type: {parser_type}")
#         documents: dict[str, list[Document]] = load_documents(source_path=source_path, parser_type=parser_type)
#         return documents

#     def _prepare_qdrant(self) -> None:
#         pass


# def create_faiss_index(embeddings, dimension):
#     res = faiss.StandardGpuResources()  # use a single GPU
#     index = faiss.IndexFlatL2(dimension)
#     gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # transfer the index to GPU
#     gpu_index.add(embeddings)
#     return gpu_index
