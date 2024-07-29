GENERIC_PROMPT_TEMPLATE = (
    "Below is the context necessary for a comprehensive understanding of the topic.\n"
    "--------------------------------------------------------\n"
    "{context_str}\n"
    "--------------------------------------------------------\n"
    "Based on the provided context, the response should accurately address the main query, incorporating relevant details and insights.\n"
    "The dialogue aims to delve into the specifics of the subject matter, highlighting key points and drawing logical conclusions.\n"
    "The response should integrate information from the context to construct a well-informed and coherent answer.\n" 
    "If certain details are not explicitly available, the response should make educated assumptions or inferences while remaining grounded in the context provided.\n"
    "The answer should be concise yet informative, offering clarity and depth to the discussion.\n"
    "Query: {query_str}\n"
    "Answer: "
)

CONTEXT_AWARE_PROMPT_TEMPLATE = (
    "Here is the relevant context which should directly guide the generation of the response:\n"
    "--------------------------------------------------------\n"
    "{context_str}\n"
    "--------------------------------------------------------\n"
    "The response must be strictly informed by and confined to the above context, ensuring it accurately addresses the main query. It should draw exclusively on the details provided, without introducing unrelated content.\n"
    "The dialogue should focus on extracting and expanding upon key elements pertinent to the subject matter, emphasizing context-driven insights and conclusions.\n"
    "The answer should leverage the context to form a coherent and well-substantiated reply. If the query involves aspects not directly mentioned in the context, the response should rely on logical inferences that remain closely aligned with the provided information.\n"
    "It is crucial that the response maintains relevance and avoids diverging into general or unrelated topics.\n"
    "Query: {query_str}\n"
    "Answer: "
)

CONTEXT_AND_LANGUAGE_AWARE_TEMPLATE = (
    "Here is the relevant context which should directly guide the generation of the response:\n"
    "--------------------------------------------------------\n"
    "{context_str}\n"
    "--------------------------------------------------------\n"
    "The response must be strictly informed by and confined to the above context, ensuring it accurately addresses the main query. It should draw exclusively on the details provided, without introducing unrelated content.\n"
    "The dialogue should focus on extracting and expanding upon key elements pertinent to the subject matter, emphasizing context-driven insights and conclusions.\n"
    "The answer should leverage the context to form a coherent and well-substantiated reply. If the query involves aspects not directly mentioned in the context, the response should rely on logical inferences that remain closely aligned with the provided information.\n"
    "The response should be formatted in the same language that the query was originally submitted in to ensure clarity and relevance.\n"
    "It is absolutely essential that the response strictly adheres to the provided context and does not deviate into general or unrelated topics. This is an urgent requirement.\n"
    "Query: {query_str}\n"
    "Answer: "
)

DOC_TEMPLATE = (
    "Here is the relevant context which should directly guide the generation of the response:\n"
    "--------------------------------------------------------\n"
    "{context_str}\n"
    "--------------------------------------------------------\n"
    "The response must be strictly informed by and confined to the above context, ensuring it accurately addresses the main query. It should draw exclusively on the details provided, without introducing unrelated content.\n"
    "The dialogue should focus on extracting and expanding upon key elements pertinent to the subject matter, emphasizing context-driven insights and conclusions.\n"
    "The answer should leverage the context to form a coherent and well-substantiated reply. If the query involves aspects not directly mentioned in the context, the response should rely on logical inferences that remain closely aligned with the provided information and should explicitly link these inferences to specific elements of the context.\n"
    "The response should be formatted in the same language and style that the query was originally submitted in to ensure clarity and relevance.\n"
    "It is absolutely essential that the response strictly adheres to the provided context and does not deviate into general or unrelated topics. This is an urgent requirement.\n"
    "If there are gaps or missing information within the provided context, the response should either explicitly acknowledge this or make cautious, well-reasoned assumptions based on the available information.\n"
    "For better readability, structure the response with appropriate paragraphs or bullet points as needed.\n"
    "Answer the query as detailed as possible with all the relevant information about all the points included in the answer.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# DOC_TEMPLATE = (
#     "Here is the relevant context which should directly guide the generation of the response:\n"
#     "--------------------------------------------------------\n"
#     "{context_str}\n"
#     "--------------------------------------------------------\n"
#     "The response must be strictly informed by and confined to the above context, ensuring it accurately addresses the main query. It should draw exclusively on the details provided, without introducing unrelated content.\n"
#     "The answer should leverage the context to form a coherent and well-substantiated reply. If the query involves aspects not directly mentioned in the context, the response should rely on logical inferences that remain closely aligned with the provided information and should explicitly link these inferences to specific elements of the context.\n"
#     "The response should be formatted in the same language and style that the query was originally submitted in to ensure clarity and relevance.\n"
#     "For better readability, structure the response with appropriate paragraphs or bullet points as needed.\n"
#     "Answer the query as detailed as possible with all the relevant information about all the points included in the answer. Don't try to summarize. Go as detailed as possible.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
