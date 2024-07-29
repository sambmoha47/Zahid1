from typing import Optional
from node import Config

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline


def get_extractors(conf: Config, llm: str) -> list:
    extractors = [
        SentenceSplitter(chunk_size=1500, chunk_overlap=300),
        QuestionsAnsweredExtractor(questions=9, llm=llm),
        TitleExtractor(nodes=4, llm=llm),
        SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
        KeywordExtractor(keywords=9, llm=llm),
    ]
    return extractors

def transform_documents(documents: dict, llm: str, conf: Optional[Config] = None) -> list:
    extractors = get_extractors(conf, llm)
    pipeline = IngestionPipeline(transformations=extractors)
    nodes = pipeline.run(documents=documents, show_progress=True)
    return nodes


