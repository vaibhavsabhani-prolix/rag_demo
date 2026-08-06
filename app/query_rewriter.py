from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from app.config import QUERY_REWRITER_MODEL


class QueryRewriter:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            QUERY_REWRITER_MODEL,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            QUERY_REWRITER_MODEL,
            trust_remote_code=True,
            device_map="auto",
        )

    def rewrite(self, query: str):
        ...