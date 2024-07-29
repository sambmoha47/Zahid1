import os
import re
import streamlit as st
from pathlib import Path
from loguru import logger
from hub import Pipeline
from utils import methods
import uuid
import time
import threading

# streamlit run src/app_streamlit.py --server.address 0.0.0.0 --server.port 8000
# ---- ---- ---- ----
# -- AUTHENTICATION
# ---- ---- ----

# List of valid credentials
VALID_CREDENTIALS = {
    "test_user": "zahid@1234",
    "noor_alamoudi": "zahid@1234",
    "hend_alkhlal": "zahid@1234",
    "bahaa_abdelsamad": "zahid@1234",
    "mohammad_zaheer": "zahid@1234",
    "mohammad_alamri": "zahid@1234",
    "asrar_ghazzawi": "zahid@1234"
}

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            if username not in st.session_state:
                st.session_state[username] = {'messages': [], 'source_docs': []}
            st.rerun()
        else:
            st.error("Invalid username or password")

# ---- ---- ---- ----
# -- CACHE START 
# ---- ---- ---- ----

@st.cache_resource
def create_pipeline():
    path_to_config = Path() / "src" / "conf" / "agent.yaml"
    path_to_secrets = Path() / "secrets"
    pipeline = Pipeline.from_conf(conf_path=path_to_config)
    pipeline.connect_client(secrets_directory=path_to_secrets)
    pipeline.prepare_settings()
    pipeline.prepare_embeddings(parser_type="llamacloud")
    pipeline.load_embeddings()
    logger.info("[CACHE INIT] Pipeline creation process completed.")
    return pipeline

glob_pipeline = create_pipeline()

# ---- ---- ---- ----
# -- MAIN APP
# ---- ---- ---- ----

def main():
    st.title("Zahid Group Policies and Procedures Chatbot")
    username = st.session_state['username']

    # Initialize session state for the user if not already done
    if username not in st.session_state:
        st.session_state[username] = {'messages': [], 'source_docs': []}

    # Initialize engine if not already done
    if 'engine' not in st.session_state:
        st.session_state.engine = glob_pipeline.spawn_query_engine(index_identifier="zahid-index")
    engine = st.session_state.engine

    # Display chat messages and source documents
    for message in st.session_state[username]['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source_docs"):
                display_source_documents(message["source_docs"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state[username]['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = prompt
            greetings = ["hello", "hi", "greetings", "hey"]
            if any(greeting in message.lower() for greeting in greetings) and len(message) < 10:
                response_text, source_docs = "Hi! What would you like to ask me about?", None
                st.markdown(response_text)
            elif 'thank' in message.lower() and len(message) < 20:
                response_text, source_docs = "I am glad to help. Let me know if you have any other question.", None
                st.markdown(response_text)
            elif glob_pipeline.check_if_user_asks_about_general_info(message):
                response_text, source_docs = display_general_info()
                st.markdown(response_text)
            elif len(message) < 10:
                response_text, source_docs = "I'd be happy to assist with your query, but I'll need a bit more information to provide a precise response. Could you please provide additional details or clarify your request?", None
                st.markdown(response_text)
            else:
                search_results = engine.query(message)
                
                # Stream the response in the main thread
                response_placeholder = st.empty()
                full_response = ""
                for chunk in search_results.response_gen:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                response_text = full_response
                source_docs = None
                if not glob_pipeline.check_if_retrieval_failed(response_text):
                    try:
                        file_pattern = r"'file_name': '(.*?)'"
                        file_names_found = re.findall(file_pattern, str(search_results.source_nodes))
                        unique_file_names = list(dict.fromkeys(file_names_found))
                        source_paths = []        
                        for file_name in unique_file_names:
                            source_paths.append(methods.find_file(directory="./data/source", filename=file_name))

                        source_docs = list(zip(unique_file_names, source_paths)) if unique_file_names else None
                        display_source_documents(source_docs)
                        st.session_state[username]['source_docs'] = source_docs
                    except Exception as error:
                        logger.error(f"[ERROR OCCURRED] {error}")
                        
        st.session_state[username]['messages'].append({"role": "assistant", "content": response_text, "source_docs": source_docs})

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()

# def main():
#     st.title("Zahid Group Policies and Procedures Chatbot")
#     username = st.session_state['username']

#     # Initialize session state for the user if not already done
#     if username not in st.session_state:
#         st.session_state[username] = {'messages': [], 'source_docs': []}

#     # Initialize engine if not already done
#     if 'engine' not in st.session_state:
#         st.session_state.engine = glob_pipeline.spawn_query_engine(index_identifier="zahid-index")
#     engine = st.session_state.engine
#     # Display chat messages and source documents
#     for message in st.session_state[username]['messages']:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and message.get("source_docs"):
#                 display_source_documents(message["source_docs"])

#     # Chat input
#     if prompt := st.chat_input("What would you like to know?"):
#         st.session_state[username]['messages'].append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             message = prompt
#             greetings = ["hello", "hi", "greetings", "hey"]
#             if any(greeting in message.lower() for greeting in greetings) and len(message) < 10:
#                 response_text, source_docs =  "Hi! What would you like to ask me about?", None
#                 st.markdown(response_text)
#             elif 'thank' in message.lower() and len(message) < 20:
#                 response_text, source_docs =  "I am glad to help. Let me know if you have any other question.", None
#                 st.markdown(response_text)
#             elif glob_pipeline.check_if_user_asks_about_general_info(message):
#                 response_text, source_docs = display_general_info()
#                 st.markdown(response_text)
#             elif len(message) < 10:
#                 response_text, source_docs = "I'd be happy to assist with your query, but I'll need a bit more information to provide a precise response. Could you please provide additional details or clarify your request?", None
#                 st.markdown(response_text)
#             else:
#                 search_results = engine.query(message)
#                 st.write_stream(search_results.response_gen)
#                 print(dir(search_results))
#                 print(search_results.get_response)

#                 response_text = search_results.response_txt
#                 if not glob_pipeline.check_if_retrieval_failed(response_text):
#                     try:
#                         file_pattern = r"'file_name': '(.*?)'"
#                         file_names_found = re.findall(file_pattern, str(search_results.source_nodes))
#                         unique_file_names = list(dict.fromkeys(file_names_found))
#                         source_paths = []        
#                         for file_name in unique_file_names:
#                             source_paths.append(methods.find_file(directory="./data/source", filename=file_name))


#                         source_docs = list(zip(unique_file_names, source_paths)) if unique_file_names else None
#                         display_source_documents(source_docs)
#                         st.session_state[username]['source_docs'] = source_docs
#                     except Exception as error:
#                         logger.error(f"[ERROR OCCURRED] {error}")
                        
#         st.session_state[username]['messages'].append({"role": "assistant", "content": response_text, "source_docs": source_docs})

#         #     print("starting msg process")
#         #     strt = time.time()
#         #     response, source_docs = process_message(prompt)
#         #     st.markdown(response)
#         #     if source_docs:
#         #         display_source_documents(source_docs)
#         #         st.session_state[username]['source_docs'] = source_docs
#         #     end = time.time()
#         #     print(f"question answered in {end-strt} sec")
#         # st.session_state[username]['messages'].append({"role": "assistant", "content": response, "source_docs": source_docs})

#     # Logout button
#     if st.sidebar.button("Logout"):
#         st.session_state['logged_in'] = False
#         st.session_state['username'] = None
#         st.rerun()

def display_source_documents(source_docs):
    st.subheader("Download Source Documents")
    for file_name, path in source_docs:
        with open(path, "rb") as file:
            st.download_button(
                label=f"Download {file_name}",
                data=file,
                file_name=file_name,
                mime="application/pdf",
                key=f"{uuid.uuid4()}"
            )

# def process_message(message):
#     # Check if the message content is a greeting
#     greetings = ["hello", "hi", "greetings", "hey"]
#     if any(greeting in message.lower() for greeting in greetings) and len(message) < 10:
#         return "Hi! What would you like to ask me about?", None
    
#     if 'thank' in message.lower() and len(message) < 20:
#         return "I am glad to help. Let me know if you have any other question.", None
    
#     strt = time.time()
#     is_general = glob_pipeline.check_if_user_asks_about_general_info(message)
#     end = time.time()
#     print("Check General", end-strt)

#     if is_general:
#         return display_general_info()

#     if len(message) < 10:
#         return "I'd be happy to assist with your query, but I'll need a bit more information to provide a precise response. Could you please provide additional details or clarify your request?", None

#     engine = st.session_state.engine
#     strt = time.time()
#     search_results = engine.query(message)
#     # search_results.print_response_stream()
#     for text in search_results.response_gen:
#         print(text, " ")
#     end = time.time()
#     print("get response", end-strt)
#     print(dir(search_results))
#     response_text = search_results.response
    
#     # strt = time.time()
#     # retrieval_failed = glob_pipeline.check_if_retrieval_failed(response_text)
#     # end = time.time()
#     # print("Check fail", end-strt)
#     retrieval_failed = False
#     if retrieval_failed:
#         return response_text, None
#     else:
#         try:
#             file_pattern = r"'file_name': '(.*?)'"
#             file_names_found = re.findall(file_pattern, str(search_results.source_nodes))
#             unique_file_names = list(dict.fromkeys(file_names_found))
#             source_paths = []        
#             for file_name in unique_file_names:
#                 source_paths.append(methods.find_file(directory="./data/source", filename=file_name))
            
#             pdf_links_text = "".join([f"\n- {pdf}" for pdf in unique_file_names])
#             # final_response_text = response_text + f"\n\nSource Documents:{pdf_links_text}\n" if unique_file_names else response_text
            
#             source_docs = list(zip(unique_file_names, source_paths)) if unique_file_names else None
            
#             return response_text, source_docs
#         except Exception as error:
#             logger.error(f"[ERROR OCCURRED] {error}")
#             return response_text, None

def display_general_info():
    directory_path = "./data/source"
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pdf"):
                file_paths.append(file_path)

    unique_file_paths = list(set(file_paths))
    pdf_links_text = "".join([f"\n- {os.path.basename(pdf)}" for pdf in unique_file_paths])
    response_text = (
        "I'm here to help answering about Zahid Group Policies and Procedures.\n\n"
        "My Cognitive abilities are limited to the information available in the source documents.\n"
        "Below you can find links to the source documents.\n\n"
    )

    final_response = response_text
    
    source_docs = [(os.path.basename(path), path) for path in unique_file_paths]
    
    return final_response, source_docs

if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        main()