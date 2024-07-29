# RAGent

## Overview
This repository contains the RAGent system, a Retrieval-Augmented Generation agent that leverages the LLaMA index and ChainLit for enhanced search and generation capabilities.

## Setup Instructions

### Prerequisites
Ensure Docker is installed on your machine to utilize containers for managing the project environment.

### Configuration Steps
1. **Secrets Configuration:**
   - Access the `secrets` folder.
   - Configure the necessary secrets for cloud integration:
     - **Vertex AI:** Populate `vertex.json` with the JSON details from Vertex AI.
     - **Azure AI Studio:** Input the necessary Azure details.
     - **Hugging Face:** Insert your token key.
     - **Llama Cloud and OpenAI Campy:** Generate and input your API key.

2. **Agent Settings:**
   - Edit `src/conf/agent.yaml` to select the desired models and clients.

### Data Management
- Store your PDF files in `data/source`. The system currently supports only PDF format and processes files recursively from this directory.

### Building and Running
1. **Build the Application:**
   - From the root directory, execute:
     ```bash
     ./build_and_run_docker.sh
     ```
2. **Application Execution:**
   - The Docker container runs the ChainLit application and listens on port `9090` as outlined in `deployment/docker/Dockerfile`.

### Local Access
- The application runs locally on `localhost:4040`.