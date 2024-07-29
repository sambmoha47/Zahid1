import os
import re
import chainlit as cl
from pathlib import Path
from loguru import logger
import os
print(os.getcwd())
from hub import Pipeline
from utils import methods


# ---- ---- ---- ----
# -- AUTH START
# ---- ---- ---- ----

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("chatbotuser", "Ch@tBot124!"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

# ---- ---- ---- ----
# -- CACHE START 
# ---- ---- ---- ----

@cl.cache
def create_pipeline():
    path_to_config = Path() / "src" / "conf" / "agent.yaml"
    path_to_secrets = Path() / "secrets"
    pipeline = Pipeline.from_conf(conf_path=path_to_config)
    pipeline.connect_client(secrets_directory=path_to_secrets)
    pipeline.prepare_settings()
    pipeline.prepare_embeddings(parser_type="llamacloud")
    logger.info("[CACHE INIT] Pipeline creation process completed.")
    return pipeline

glob_pipeline = create_pipeline()

# ---- ---- ---- ----
# ---- ON CHAT START
# ---- ---- ---- ----

@cl.on_chat_start
async def factory():
    logger.info("[CHAT ACTIVATION] Chat session initiated.")
    engine = glob_pipeline.spawn_query_engine(index_identifier="zahid-index")
    cl.user_session.set("engine", engine)
    await cl.Message(
        author="Assistant", content="Hello! How can I assist you today?", elements=[],
    ).send()

# ---- ---- ---- ----
# ---- ON USER MSG
# ---- ---- ---- ----

@cl.on_message
async def process_message(message: cl.Message):
    # Check if the message content is a greeting
    greetings = ["hello", "hi", "greetings", "hey"]
    if any(greeting in message.content.lower() for greeting in greetings) and len(message.content) < 10:
        # If it's a greeting, send a simple message without elements
        await cl.Message(content="Hi! What would you like to ask me about?", elements=[]).send()
        return

    is_general = glob_pipeline.check_if_user_asks_about_general_info(message.content)
    if is_general:
        # Recursively scan the directory and prepare elements for each unique file
        directory_path = "./data/source"
        file_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(".pdf"):  # Assuming you're looking for PDF files
                    file_paths.append(file_path)

        # Ensure all file paths are unique
        unique_file_paths = list(set(file_paths))

        # Use basename for each path in the pdf_links_text
        pdf_links_text = "".join([f"\n{os.path.basename(pdf)}" for pdf in unique_file_paths])
        response_text = (
            "I'm here to help answering about Zahid Group Policies and Procedures.\n\n"
            "My Cognitive abbilities are limited to the information available in the source documents.\n"
            "Attached to the response please find source documents.\n\n"
        )

        final_response = response_text + f"\nSource Documents:{pdf_links_text}\n"
        await cl.Message(content=final_response).send()
        return

    if len(message.content) < 10:
        await cl.Message(content="I'd be happy to assist with your query, but I'll need a bit more information to provide a precise response.\n\
            Could you please provide additional details or clarify your request?", 
            elements=[]).send()
        return

    engine = cl.user_session.get("engine")
    search_results = engine.query(message.content)
    response_text = search_results.response
    retrieval_failed = glob_pipeline.check_if_retrieval_failed(response_text)
    response_elements = []
    if retrieval_failed:
        final_response_text = response_text
    else:
        try:
            # Regex pattern to extract 'file_name'
            file_pattern = r"'file_name': '(.*?)'"
            file_names_found = re.findall(file_pattern, str(search_results.source_nodes))

            # Remove duplicate file names
            unique_file_names = list(dict.fromkeys(file_names_found))
            # Constructing paths for the found source files
            source_paths = []        
            for file_name in unique_file_names:
                source_paths.append(methods.find_file(directory="./data/source", filename=file_name))
            # Creating PDF elements for the response
            response_elements = [
                cl.Pdf(name=file_name, display="side", path=path)
                for file_name, path in zip(unique_file_names, source_paths)
            ]
            # Enhancing the response text with links to the associated PDFs
            pdf_links_text = "".join([f"\n{pdf}" for pdf in unique_file_names])
            if response_elements:
                final_response_text = response_text + f"\n\nSource Documents:{pdf_links_text}\n"
            else: 
                final_response_text = response_text
        except Exception as error:
            response_elements = []
            final_response_text = response_text
            logger.error(f"[ERROR OCCURRED] {error}")

    await cl.Message(content=final_response_text, elements=response_elements).send()