from dataclasses import dataclass, field
from typing import Dict

from loguru import logger

@dataclass
class UserMapping:
    mapping: Dict[str, str] = field(default_factory=lambda: {
        "Jakub": "medical",
        "User": "commercial"
    })

    def get_category(self, name: str) -> str:
        """ Return the category based on the provided name. """
        return self.mapping.get(name, "commercial")  # Return 'commercial' if name is not in the mapping
