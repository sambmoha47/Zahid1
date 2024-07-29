# Storage

based on: 
1. https://medium.com/how-ai-built-this/zero-to-one-a-guide-to-building-a-first-pdf-chatbot-with-langchain-llamaindex-part-1-7d0e9c0d62f
2. https://betterprogramming.pub/llamaindex-how-to-use-index-correctly-6f928b8944c6
3. https://medium.com/p/99ca9851be9c


Index Storage: LLM models often require indexing mechanisms to optimize search and retrieval operations.
Index storage systems, such as Langchain or LlamaIndex, provide efficient indexing capabilities specifically designed for LLM applications.
These systems enable faster query processing and facilitate advanced search functionalities.

Vector Storage: LLM models often represent text inputs as dense vectors in high-dimensional spaces.
Efficient storage and retrieval of these vectors are critical for similarity search, clustering, and other advanced operations.
Vector storage systems, like VectorDB or DeepStorage, offer specialized support for storing and querying high-dimensional vectors,
enabling efficient processing of LLM-related tasks.

- Traditional storage like SQL or NoSQL is not efficient to perform searches on large chunks of text with similar meanings.
- Vector Storage such as ChromaDB or Pincone will store your embedding data and Index Storage will be used to store indexing of those embeddings data

# Index
```
- Nodes
  - Node 1
    - Text content
    - Metadata
    - Relationships
  - Node 2
    - Text content
    - Metadata
    - Relationships
  - ...
- StorageContext
  - IndexStore
    - Index metadata
    - Index data structures
  - VectorStore
    - Vector representations of nodes
  - DocumentStore
    - Original document data
- Settings
  - Chunk size limit
  - Max input size
  - Embedding model
  - LLM model
  - ...
```

Explanation:

1. The Index is the top-level structure that contains all the components.

2. Nodes represent the granular units of indexable data.
   - Each Node has its own text content, metadata, and relationships with other nodes.
   - Nodes are created by parsing Documents or manually creating them.

3. StorageContext manages the storage and persistence of the index data.
   - IndexStore holds the index metadata and data structures for efficient querying.
   - VectorStore stores the vector representations of the nodes for similarity search.
   - DocumentStore keeps the original document data for reference.

4. Settings contain the configuration options for the index.
   - Chunk size limit controls the maximum size of each node.
   - Max input size sets the maximum input size for the LLM.
   - Embedding model specifies the model used for generating vector representations.
   - LLM model specifies the language model used for query processing.
   - Other settings can be added as needed.

This structure allows for a modular and flexible design, where each component can be customized and extended as needed. The Nodes provide the core indexable data, the StorageContext manages the storage and persistence, and the Settings control the behavior and configuration of the index.