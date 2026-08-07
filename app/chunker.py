"""
Production Patent Chunker V3

Token-aware semantic chunking pipeline.

Pipeline:

    Document
        ↓
    Section Detection       (dynamic heading detection)
        ↓
    Semantic Unit Detection (paragraphs → sentences → word fragments)
        ↓
    Token-aware Chunk Builder (greedy merge, no overlap)
        ↓
    Chunk Validation        (reject empty / duplicate only)
        ↓
    PatentChunk objects

Each section is processed independently. No chunk ever contains
content from another section.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.chunking.token_counter import TokenCounter
from app.chunking.section_detector import SectionDetector
from app.chunking.semantic_unit_splitter import SemanticUnitSplitter
from app.chunking.chunk_builder import ChunkBuilder
from app.chunking.chunk_validator import ChunkValidator
from app.config import MAX_CHUNK_TOKENS, DEBUG_CHUNKS, DEBUG_CHUNKS_DIR
from app.models.patent_document import PatentDocument
from app.models.patent_chunk import PatentChunk


class PatentChunker:
    """
    Production-ready document chunker.

    Orchestrates the full chunking pipeline:

    1. SectionDetector      — splits document into headed sections
    2. SemanticUnitSplitter  — breaks sections into semantic units
    3. ChunkBuilder          — merges units into token-bounded chunks
    4. ChunkValidator        — rejects empty / duplicate chunks

    Each section is processed as an independent stream.
    No chunk ever contains content from another section.

    Usage::

        chunker = PatentChunker()
        chunks = chunker.split(document)
    """

    def __init__(
        self,
        max_tokens: int = MAX_CHUNK_TOKENS,
        debug: bool = DEBUG_CHUNKS,
        debug_dir: str = DEBUG_CHUNKS_DIR,
    ) -> None:

        self._max_tokens = max_tokens
        self._debug = debug
        self._debug_dir = debug_dir

        # Shared tokenizer instance
        self.token_counter = TokenCounter()

        # Pipeline components
        self.section_detector = SectionDetector()

        self.unit_splitter = SemanticUnitSplitter(
            token_counter=self.token_counter,
            max_tokens=max_tokens,
        )

        self.chunk_builder = ChunkBuilder(
            token_counter=self.token_counter,
            max_tokens=max_tokens,
        )

        self.chunk_validator = ChunkValidator(
            token_counter=self.token_counter,
        )

    # ==============================================================
    # Main entry point
    # ==============================================================

    def split(
        self,
        document: PatentDocument,
    ) -> list[PatentChunk]:
        """
        Split a PatentDocument into validated, token-bounded chunks.

        This is the only method downstream code needs to call.
        The signature is fully backward-compatible with V2.
        """

        # Reset duplicate tracker for each new document
        self.chunk_validator.reset()

        # Clear token counter cache between documents to bound memory
        self.token_counter.clear_cache()

        patent_chunks: list[PatentChunk] = []
        chunk_index = 1

        # ---- Step 1: Section Detection ----
        sections = self.section_detector.detect(document.text)

        total_sections = len(sections)

        # Pre-compute section chunk counts for metadata.
        # We build all sections first, then do a lightweight post-pass
        # to fill in total_chunks. This avoids redesigning the pipeline.
        section_chunk_counts: dict[str, int] = {}

        # ---- Process each section independently ----
        for section in sections:

            # ---- Step 2: Semantic Unit Detection ----
            units = self.unit_splitter.split(section.content)

            if not units:
                continue

            # ---- Step 3: Token-aware Chunk Building ----
            built_chunks = self.chunk_builder.build(
                section_heading=section.heading,
                units=units,
            )

            # Track how many valid chunks this section produces
            section_valid_count = 0

            # ---- Step 4: Validate and produce PatentChunks ----

            for built in built_chunks:

                if not self.chunk_validator.is_valid(built.text):
                    continue

                section_valid_count += 1

                patent_chunks.append(
                    PatentChunk(
                        patent_id=document.patent_id,
                        chunk_id=chunk_index,
                        section=built.section,
                        text=built.text,
                        token_count=built.token_count,
                        word_count=built.word_count,
                        document_chunk_index=chunk_index,
                        section_chunk_index=section_valid_count,
                        total_sections=total_sections,
                        chunk_uuid="",  # filled below after point_id is set
                    )
                )

                chunk_index += 1

            # Record section chunk count
            if section_valid_count > 0:
                section_chunk_counts[section.heading] = section_valid_count

        # ---- Post-pass: fill in total_chunks and section_total_chunks ----
        # This is a lightweight O(n) pass over the already-built list.
        # No tokenization, no re-processing — just metadata assignment.
        total_chunks = len(patent_chunks)

        for chunk in patent_chunks:
            chunk.total_chunks = total_chunks
            chunk.section_total_chunks = section_chunk_counts.get(
                chunk.section, 0,
            )
            # Set chunk_uuid to match point_id for clarity
            chunk.chunk_uuid = chunk.point_id

        # ---- Optional: Debug output ----
        if self._debug:
            self._write_debug_output(document.patent_id, patent_chunks)

        return patent_chunks

    # ==============================================================
    # Debug output
    # ==============================================================

    def _write_debug_output(
        self,
        patent_id: str,
        chunks: list[PatentChunk],
    ) -> None:
        """
        Write each chunk to a text file for inspection.

        Output goes to: debug_dir/patent_id/chunk_001.txt
        Only runs when DEBUG_CHUNKS is True. Does not affect
        production performance when disabled.
        """

        debug_path = Path(self._debug_dir) / patent_id
        debug_path.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            filename = f"chunk_{chunk.chunk_id:04d}.txt"
            filepath = debug_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Patent ID          : {chunk.patent_id}\n")
                f.write(f"Section            : {chunk.section}\n")
                f.write(f"Chunk              : {chunk.section_chunk_index}"
                        f" of {chunk.section_total_chunks}\n")
                f.write(f"Document Chunk     : {chunk.document_chunk_index}"
                        f" of {chunk.total_chunks}\n")
                f.write(f"Token Count        : {chunk.token_count}\n")
                f.write(f"Word Count         : {chunk.word_count}\n")
                f.write(f"Chunk UUID         : {chunk.chunk_uuid}\n")
                f.write(f"Created At         : {chunk.created_at}\n")
                f.write(f"\n{'=' * 60}\n\n")
                f.write(chunk.text)
                f.write("\n")

    # ==============================================================
    # Statistics
    # ==============================================================

    def print_statistics(
        self,
        chunks: list[PatentChunk],
    ) -> None:
        """
        Print chunk statistics.
        """

        if not chunks:
            print("No chunks created.")
            return

        token_sizes = [c.token_count for c in chunks]
        word_sizes = [c.word_count for c in chunks]

        # Collect unique sections
        sections = {c.section for c in chunks}

        print()
        print("=" * 60)
        print("Chunk Statistics")
        print("=" * 60)
        print(f"  Total Chunks    : {len(chunks)}")
        print(f"  Sections Found  : {len(sections)}")
        print()
        print("  Token Counts:")
        print(f"    Largest       : {max(token_sizes)}")
        print(f"    Smallest      : {min(token_sizes)}")
        print(f"    Average       : {sum(token_sizes) // len(token_sizes)}")
        print()
        print("  Word Counts:")
        print(f"    Largest       : {max(word_sizes)}")
        print(f"    Smallest      : {min(word_sizes)}")
        print(f"    Average       : {sum(word_sizes) // len(word_sizes)}")
        print()
        print("  Sections:")

        for section_name in sorted(sections):
            count = sum(1 for c in chunks if c.section == section_name)
            print(f"    {section_name}: {count} chunks")

        print("=" * 60)
