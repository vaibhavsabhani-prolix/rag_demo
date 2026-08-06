from dataclasses import dataclass


@dataclass
class PatentDocument:

    patent_id: str

    text: str

    metadata: dict