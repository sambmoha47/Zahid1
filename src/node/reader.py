import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from loguru import logger

@dataclass
class IndexConfig:
    path: str
    index_type: str
    load_existing_index_under_prefix: bool
    single_index_name: Optional[str] = None
    folder_indexes: Optional[List[Dict[str, str]]] = None

@dataclass
class ModelConfig:
    self_hosted: bool
    client: str
    prefix: str
    hyperparameters: Dict[str, int]

@dataclass
class ClientConfig:
    identifier: str
    description: str
    models: Dict[str, ModelConfig]

@dataclass
class ParserConfig:
    identifier: str
    description: str
    splitters: Dict[str, Any]
    extractors: Dict[str, Any]

@dataclass
class Config:
    index: IndexConfig
    client: ClientConfig
    parser: ParserConfig

def read_configuration(yaml_file_path: Path) -> Config:
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data = yaml_data['fields']

    index = IndexConfig(
        path=yaml_data['index']['path'],
        index_type=yaml_data['index']['type'],
        load_existing_index_under_prefix=yaml_data['index']['load_existing_index_under_prefix'],
        single_index_name=yaml_data['index'].get('single_index_name'),
        folder_indexes=yaml_data['index'].get('folder_indexes')
    )

    client_yaml_path = Path("src/conf/protocol/client") / yaml_data['protocol']['client']
    with open(client_yaml_path, 'r') as file:
        client_config = yaml.safe_load(file)
    client_models = {
        model_name: ModelConfig(
            self_hosted=model_details['self_hosted'],
            client=model_details['client'],
            prefix=model_details['prefix'],
            hyperparameters=model_details.get('hyperparameters', dict())
        ) for model_name, model_details in client_config['fields']['models'].items()
    }
    client = ClientConfig(
        identifier=client_config['identifier'],
        description=client_config['description'],
        models=client_models
    )

    parser_yaml_path = Path("src/conf/protocol/parser") / yaml_data['protocol']['parser']
    with open(parser_yaml_path, 'r') as file:
        parser_config = yaml.safe_load(file)
    parser = ParserConfig(
        identifier=parser_config['identifier'],
        description=parser_config['description'],
        splitters=parser_config['fields']['splitters'],
        extractors=parser_config['fields']['extractors']
    )
    configuration = Config(index=index, client=client, parser=parser)

    formatted_config = (
        f"{100*'-'}\n"
        f"Index: {vars(configuration.index)}\n"
        f"Client: {vars(configuration.client)}\n"
        f"Parser: {vars(configuration.parser)}\n"
        f"{100*'-'}\n"
    )
    formatted_config = formatted_config.replace("{", "").replace("}", "")
    logger.info(f"[Configuration] Configuration loaded from {yaml_file_path}:\n{formatted_config}")

    return configuration
