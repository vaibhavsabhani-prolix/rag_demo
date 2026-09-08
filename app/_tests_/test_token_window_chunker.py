"""
Comprehensive Unit Tests for TokenWindowChunker and PatentChunker.

Covers all 21 required test scenarios:
1. Section smaller than 4096 tokens
2. Section exactly 4096 tokens
3. Section slightly larger than 4096 tokens
4. 15,000-token section
5. Multiple sections (strict section isolation)
6. Chunk boundary inside a word
7. Chunk boundary inside a sentence
8. Paragraph boundary near target
9. Sentence boundary near target
10. No whitespace / CJK text
11. Very long word (unbounded word fallback)
12. Empty section
13. Section containing only whitespace
14. Multiple paragraphs
15. Abbreviations such as Dr. / Fig.
16. Decimal numbers such as 3.14
17. Closing quotes/brackets
18. Deterministic chunk point IDs
19. Correct section_chunk_index
20. Correct total_chunks
21. Correct section_total_chunks
And verifies: all(chunk.token_count <= MAX_CHUNK_TOKENS for chunk in chunks)
"""

import unittest
from pathlib import Path

from app.chunker import PatentChunker, _chunk_point_id
from app.chunking.token_counter import TokenCounter
from app.chunking.token_window_chunker import TokenWindowChunker
from app.config import MAX_CHUNK_TOKENS
from app.models.patent_document import PatentDocument


class TestTokenWindowChunker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.token_counter = TokenCounter()
        cls.chunker = PatentChunker(max_tokens=MAX_CHUNK_TOKENS)

    # ------------------------------------------------------------------
    # 1. Section smaller than 4096 tokens
    # ------------------------------------------------------------------
    def test_01_section_smaller_than_4096(self):
        text = "This is a small section of text. It has only two sentences."
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Abstract", text)
        self.assertEqual(len(chunks), 1)
        self.assertLess(chunks[0].token_count, 4096)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].section, "Abstract")

    # ------------------------------------------------------------------
    # 2. Section exactly 4096 tokens
    # ------------------------------------------------------------------
    def test_02_section_exactly_4096(self):
        # Create a text with exactly 4096 tokens
        base_sentence = "The present invention relates to an apparatus for processing data. "
        base_tokens = self.token_counter.count(base_sentence)
        reps = 4096 // base_tokens
        text = base_sentence * reps
        # Pad or trim to exactly 4096 tokens
        input_ids, offsets = self.token_counter.tokenize_with_offsets(text)
        if len(input_ids) < 4096:
            # pad with words
            extra_needed = 4096 - len(input_ids)
            text += " word" * extra_needed
            input_ids, offsets = self.token_counter.tokenize_with_offsets(text)
        exact_text = text[: offsets[4095][1]]
        exact_tokens = self.token_counter.count(exact_text)
        self.assertEqual(exact_tokens, 4096)

        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Description", exact_text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].token_count, 4096)
        self.assertTrue(all(c.token_count <= 4096 for c in chunks))

    # ------------------------------------------------------------------
    # 3. Section slightly larger than 4096 tokens
    # ------------------------------------------------------------------
    def test_03_section_slightly_larger_than_4096(self):
        base_sentence = "The apparatus comprises a processor and a memory coupled to the processor. "
        text = ""
        while self.token_counter.count(text) <= 4096:
            text += base_sentence
        total_tokens = self.token_counter.count(text)
        self.assertGreater(total_tokens, 4096)
        self.assertLess(total_tokens, 4200)

        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Description", text)
        self.assertEqual(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(c.token_count, 4096)

    # ------------------------------------------------------------------
    # 4. 15000-token section
    # ------------------------------------------------------------------
    def test_04_15000_token_section(self):
        sentence = "A system and method for distributed patent analysis using machine learning. "
        block = sentence * 10 + "\n\n"
        text = ""
        while self.token_counter.count(text) < 15000:
            text += block
        total_tokens = self.token_counter.count(text)
        self.assertGreaterEqual(total_tokens, 15000)

        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Detailed Description", text)
        self.assertGreaterEqual(len(chunks), 4)
        for i, c in enumerate(chunks):
            self.assertLessEqual(
                c.token_count,
                4096,
                f"Chunk {i} exceeded 4096 tokens: {c.token_count}",
            )

    # ------------------------------------------------------------------
    # 5. Multiple sections (section isolation)
    # ------------------------------------------------------------------
    def test_05_multiple_sections(self):
        doc_text = (
            "ABSTRACT:\n"
            "This is the abstract text describing the invention.\n\n"
            "FIELD OF THE INVENTION:\n"
            "This invention relates to neural networks.\n\n"
            "DETAILED DESCRIPTION:\n"
            "Detailed explanation of embodiments."
        )
        doc = PatentDocument(
            patent_id="TEST001",
            text=doc_text,
            metadata={},
        )
        chunks = self.chunker.split(doc)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].section, "ABSTRACT")
        self.assertEqual(chunks[1].section, "FIELD OF THE INVENTION")
        self.assertEqual(chunks[2].section, "DETAILED DESCRIPTION")

        # Verify no cross-section leakage
        self.assertIn("abstract text", chunks[0].text)
        self.assertNotIn("neural networks", chunks[0].text)
        self.assertIn("neural networks", chunks[1].text)
        self.assertNotIn("Detailed explanation", chunks[1].text)

    # ------------------------------------------------------------------
    # 6. Chunk boundary inside a word
    # ------------------------------------------------------------------
    def test_06_chunk_boundary_inside_a_word(self):
        # Using a small max_tokens to test boundary falling inside a multi-token word
        # "vaibhav" tokenizes to 4 tokens: [' va', 'ib', 'h', 'av']
        text = "hello my name is vaibhav. i am fine."
        # Find token index in the middle of 'vaibhav'
        input_ids, offsets = self.token_counter.tokenize_with_offsets(text)
        # Token 4 starts ' va', token 5 is 'ib'
        # With max_tokens=5, nominal target lands on 'ib' (token 5)
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=5,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertTrue(all(c.token_count <= 5 for c in chunks))
        # Word 'vaibhav' should NOT be split across chunks as 'vaibh' and 'av'
        for c in chunks:
            self.assertFalse(
                c.text.endswith("vaibh") or c.text.endswith("va") or c.text.startswith("av")
            )

    # ------------------------------------------------------------------
    # 7. Chunk boundary inside a sentence
    # ------------------------------------------------------------------
    def test_07_chunk_boundary_inside_a_sentence(self):
        s1 = "This is the first complete sentence."
        s2 = "This is the second complete sentence."
        text = f"{s1} {s2}"
        s1_tokens = self.token_counter.count(s1)
        s2_tokens = self.token_counter.count(s2)
        max_t = max(s1_tokens, s2_tokens) + 2
        self.assertGreater(self.token_counter.count(text), max_t)

        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=max_t,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertTrue(all(c.token_count <= max_t for c in chunks))
        # Chunk 1 should end cleanly at sentence 1 boundary
        self.assertEqual(chunks[0].text, s1)
        self.assertEqual(chunks[1].text, s2)

    # ------------------------------------------------------------------
    # 8. Paragraph boundary near target (prefers paragraph over sentence)
    # ------------------------------------------------------------------
    def test_08_paragraph_boundary_near_target(self):
        p1 = "First paragraph sentence one. First paragraph sentence two."
        p2 = "Second paragraph sentence one. Second paragraph sentence two."
        text = f"{p1}\n\n{p2}"
        p1_tokens = self.token_counter.count(p1)
        # Set max_tokens to slightly exceed p1
        max_t = p1_tokens + 5
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=max_t,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertTrue(all(c.token_count <= max_t for c in chunks))
        self.assertEqual(chunks[0].text, p1)
        self.assertEqual(chunks[1].text, p2)

    # ------------------------------------------------------------------
    # 9. Sentence boundary near target (prefers sentence over word)
    # ------------------------------------------------------------------
    def test_09_sentence_boundary_near_target(self):
        s1 = "Alpha sentence is complete."
        s2 = "Beta sentence is also complete."
        s3 = "Gamma sentence begins here and continues."
        text = f"{s1} {s2} {s3}"
        s1_s2_tokens = self.token_counter.count(f"{s1} {s2}")
        max_t = s1_s2_tokens + 3
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=max_t,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertTrue(all(c.token_count <= max_t for c in chunks))
        self.assertEqual(chunks[0].text, f"{s1} {s2}")

    # ------------------------------------------------------------------
    # 10. No whitespace / CJK text
    # ------------------------------------------------------------------
    def test_10_no_whitespace_cjk_text(self):
        cjk_text = (
            "本发明涉及一种新型专利技术。包括以下特征：提高系统效率。"
            "同时降低能源消耗。在第二方面，该装置还具备自动诊断功能。"
            "该诊断功能可实时监控硬件工作状态。"
        )
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=30,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("CJK Section", cjk_text)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(c.token_count, 30)
            self.assertTrue(len(c.text) > 0)
        # Verify text continuity (no lost characters)
        rejoined = "".join(c.text for c in chunks)
        # Spaces might be stripped, but CJK has no spaces so rejoined should match
        self.assertEqual(rejoined, cjk_text)

    # ------------------------------------------------------------------
    # 11. Very long word (unbounded word fallback)
    # ------------------------------------------------------------------
    def test_11_very_long_word(self):
        # 300-character single word without whitespace
        long_word = "A" * 300
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=20,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", long_word)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, 20)
        rejoined = "".join(c.text for c in chunks)
        self.assertEqual(rejoined, long_word)

    # ------------------------------------------------------------------
    # 12. Empty section
    # ------------------------------------------------------------------
    def test_12_empty_section(self):
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Empty", "")
        self.assertEqual(len(chunks), 0)

    # ------------------------------------------------------------------
    # 13. Section containing only whitespace
    # ------------------------------------------------------------------
    def test_13_section_containing_only_whitespace(self):
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("Whitespace", "   \n\n\t  \n  ")
        self.assertEqual(len(chunks), 0)

    # ------------------------------------------------------------------
    # 14. Multiple paragraphs
    # ------------------------------------------------------------------
    def test_14_multiple_paragraphs(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=4096,
        )
        chunks = window_chunker.chunk("MultiPara", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    # ------------------------------------------------------------------
    # 15. Abbreviations such as Dr. / Fig.
    # ------------------------------------------------------------------
    def test_15_abbreviations(self):
        text = "Dr. Smith presented Fig. 3 in the meeting. The data was verified."
        tokens = self.token_counter.count(text)
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=tokens,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertEqual(len(chunks), 1)
        # If we cut with max_tokens slightly smaller than full text:
        dr_tokens = self.token_counter.count("Dr. Smith presented Fig. 3")
        window_chunker_small = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=dr_tokens + 2,
            search_window_ratio=0.5,
        )
        chunks_small = window_chunker_small.chunk("Doc", text)
        # Should not split immediately after Dr. or Fig.
        for c in chunks_small:
            self.assertFalse(c.text.endswith("Dr.") or c.text.endswith("Fig."))

    # ------------------------------------------------------------------
    # 16. Decimal numbers such as 3.14
    # ------------------------------------------------------------------
    def test_16_decimal_numbers(self):
        text = "The value of pi is approximately 3.14159 in this formula. Next sentence."
        pi_tokens = self.token_counter.count("The value of pi is approximately 3.14159")
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=pi_tokens + 2,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        # Should not split inside 3.14159
        for c in chunks:
            self.assertFalse(c.text.endswith("3.") or c.text.startswith("14159"))

    # ------------------------------------------------------------------
    # 17. Closing quotes/brackets
    # ------------------------------------------------------------------
    def test_17_closing_quotes_and_brackets(self):
        text = 'He said "Stop!" Then he walked away.'
        quote_tokens = self.token_counter.count('He said "Stop!"')
        window_chunker = TokenWindowChunker(
            token_counter=self.token_counter,
            max_tokens=quote_tokens + 1,
            search_window_ratio=0.5,
        )
        chunks = window_chunker.chunk("Doc", text)
        self.assertEqual(chunks[0].text, 'He said "Stop!"')
        self.assertEqual(chunks[1].text, "Then he walked away.")

    # ------------------------------------------------------------------
    # 18. Deterministic chunk point IDs
    # ------------------------------------------------------------------
    def test_18_deterministic_chunk_point_ids(self):
        doc = PatentDocument(
            patent_id="DET001",
            text="SECTION 1:\nFirst section content.\n\nSECTION 2:\nSecond section content.",
            metadata={},
        )
        chunks_run1 = self.chunker.split(doc)
        chunks_run2 = self.chunker.split(doc)
        self.assertEqual(len(chunks_run1), len(chunks_run2))
        for c1, c2 in zip(chunks_run1, chunks_run2):
            self.assertEqual(c1.point_id, c2.point_id)
            self.assertEqual(c1.chunk_uuid, c2.chunk_uuid)
            self.assertEqual(
                c1.point_id,
                _chunk_point_id("DET001", c1.chunk_id),
            )

    # ------------------------------------------------------------------
    # 19. Correct section_chunk_index
    # ------------------------------------------------------------------
    def test_19_correct_section_chunk_index(self):
        # Construct two sections, where section 1 has multiple chunks and section 2 has 1 chunk
        s1_text = "".join(f"Unique sentence number {i} in section one.\n\n" for i in range(30))
        s2_text = "Short section two."
        doc = PatentDocument(
            patent_id="INDEX001",
            text=f"SECTION A:\n{s1_text}\nSECTION B:\n{s2_text}",
            metadata={},
        )
        small_chunker = PatentChunker(max_tokens=100)
        chunks = small_chunker.split(doc)
        sec_a_chunks = [c for c in chunks if c.section == "SECTION A"]
        sec_b_chunks = [c for c in chunks if c.section == "SECTION B"]
        self.assertGreater(len(sec_a_chunks), 1)
        self.assertEqual(len(sec_b_chunks), 1)

        # section_chunk_index must be 1, 2, 3... within its section
        for i, c in enumerate(sec_a_chunks, start=1):
            self.assertEqual(c.section_chunk_index, i)
        self.assertEqual(sec_b_chunks[0].section_chunk_index, 1)

    # ------------------------------------------------------------------
    # 20. Correct total_chunks
    # ------------------------------------------------------------------
    def test_20_correct_total_chunks(self):
        doc = PatentDocument(
            patent_id="TOTAL001",
            text="SECTION 1:\nSentence 1.\n\nSECTION 2:\nSentence 2.",
            metadata={},
        )
        chunks = self.chunker.split(doc)
        total = len(chunks)
        self.assertEqual(total, 2)
        for c in chunks:
            self.assertEqual(c.total_chunks, total)

    # ------------------------------------------------------------------
    # 21. Correct section_total_chunks
    # ------------------------------------------------------------------
    def test_21_correct_section_total_chunks(self):
        s1_text = "".join(f"Unique sentence number {i} in section one.\n\n" for i in range(30))
        s2_text = "Short section two."
        doc = PatentDocument(
            patent_id="SEC_TOTAL001",
            text=f"SECTION A:\n{s1_text}\nSECTION B:\n{s2_text}",
            metadata={},
        )
        small_chunker = PatentChunker(max_tokens=100)
        chunks = small_chunker.split(doc)
        sec_a_count = sum(1 for c in chunks if c.section == "SECTION A")
        sec_b_count = sum(1 for c in chunks if c.section == "SECTION B")

        for c in chunks:
            if c.section == "SECTION A":
                self.assertEqual(c.section_total_chunks, sec_a_count)
            elif c.section == "SECTION B":
                self.assertEqual(c.section_total_chunks, sec_b_count)

    # ------------------------------------------------------------------
    # 22. Real patent regression validation
    # ------------------------------------------------------------------
    def test_22_real_patents_regression(self):
        candidate_paths = [
            Path("patents-processed/AP170S1.txt"),
            Path("Book-cover-list/ATE269222T1.txt"),
            Path("patents-processed/JP7363836B2.txt"),
            Path("Book-cover-list/CN216298736U.txt"),
        ]
        from app.parser import PatentParser

        parser = PatentParser()
        tested = 0
        for p in candidate_paths:
            if not p.exists():
                continue
            doc = parser.load_patent(p)
            chunks = self.chunker.split(doc)
            self.assertGreater(len(chunks), 0, f"No chunks for {p.name}")
            for c in chunks:
                self.assertLessEqual(
                    c.token_count,
                    MAX_CHUNK_TOKENS,
                    f"Chunk {c.chunk_id} in {p.name} exceeded MAX_CHUNK_TOKENS: {c.token_count}",
                )
                self.assertGreater(len(c.text), 0)
                self.assertEqual(c.patent_id, doc.patent_id)
            tested += 1
        self.assertGreater(tested, 0, "No real patent files were found to test.")


if __name__ == "__main__":
    unittest.main()
