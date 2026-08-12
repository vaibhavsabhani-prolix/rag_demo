"""
Patent File Parser
"""

import json
from pathlib import Path
from app.models.patent_document import PatentDocument


class PatentParser:

    def read_text(self, txt_path: Path) -> str:
        # Read patent text file.
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def read_metadata(self, json_path: Path) -> dict:
        # Read patent metadata.
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def load_patent(self, txt_path: Path) -> PatentDocument:
        # Read TXT and JSON files and return a PatentDocument.
        json_path = txt_path.with_suffix(".json")

        text = self.read_text(txt_path)

        metadata = self.read_metadata(json_path)

        return PatentDocument(
            patent_id=txt_path.stem,
            text=text,
            metadata=metadata,
        )
