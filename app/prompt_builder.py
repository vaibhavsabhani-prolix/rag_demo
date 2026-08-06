"""
Prompt Builder

Builds LLM prompts from patent-level search results.
Each patent contributes its best chunk + any additional
matching chunks as context.
"""

from __future__ import annotations

from app.models.patent_search_result import PatentSearchResult


class PromptBuilder:

    def build(
        self,
        question: str,
        search_results: list[PatentSearchResult],
    ) -> str:
        """
        Build a prompt for the LLM using patent-level search results.

        Each patent's matching chunks are included as context,
        ordered by reranker score (best first).
        """

        prompt = []

        prompt.append(
            "You are an expert patent assistant."
        )

        prompt.append(
            "Answer ONLY using the provided patent context."
        )

        prompt.append(
            "If the answer cannot be found in the context, say you don't know."
        )

        prompt.append("\n================ CONTEXT ================\n")

        for index, patent in enumerate(search_results, start=1):

            prompt.append(
                f"Patent {index}\n"
                f"Patent ID : {patent.patent_id}\n"
                f"Score     : {patent.score:.4f}\n"
            )

            for chunk_idx, chunk in enumerate(patent.matching_chunks, start=1):
                prompt.append(
                    f"  Chunk {chunk_idx} (ID: {chunk.chunk_id}, "
                    f"Section: {chunk.section}):\n"
                    f"{chunk.text}\n"
                )

            prompt.append("-----------------------------------------\n")

        prompt.append("\n================ QUESTION ================\n")

        prompt.append(question)

        prompt.append("\n================ ANSWER ================\n")

        return "\n".join(prompt)