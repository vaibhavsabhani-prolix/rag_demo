# Patent Indexing — Parsing & Chunking Flow

## 1. Take Patent from `PATENT_DIRECTORY`

```text
PATENT_DIRECTORY
    ↓
Take one patent file
    ↓
Example: US10001.txt
```

---

## 2. Parse the Patent and Detect Sections

The `PatentParser` reads the patent and detects its sections based on their headings.

```text
Patent
    ↓
Detect sections
    ↓
Section 1
Section 2
Section 3
...
```

Each section is represented approximately as:

```python
Section(
    heading="title-english",
    content="LED TV",
    start_offset=0,
    end_offset=6,
)
```

Example:

```text
Patent
│
├── Section 1
│   ├── heading = "Title"
│   └── content = "LED TV"
│
├── Section 2
│   ├── heading = "Abstract"
│   └── content = "The present invention..."
│
├── Section 3
│   ├── heading = "Background"
│   └── content = "..."
│
└── Section 4
    ├── heading = "Claims"
    └── content = "..."
```

---

# 3. Take One Section

The `PatentChunker` processes each section separately.

```text
Section
    ↓
Take section content
    ↓
Split content into paragraphs
```

Example:

```text
Section: Detailed Description

    Paragraph 1
    Paragraph 2
    Paragraph 3
    Paragraph 4
```

---

# 4. Take One Paragraph

The chunker processes the paragraphs one by one.

```text
Paragraph 1
    ↓
Check paragraph size
```

Assume:

```text
MAX_TOKENS = 512
CERTAINLY_OVERSIZED_CHARS_PER_TOKEN = 8
```

First check:

```text
paragraph characters
        >
512 × 8
```

If the paragraph is larger than this oversized-character threshold:

```text
Paragraph
    ↓
Split into sentences
```

Otherwise, count the actual tokens.

---

# 5. Count Paragraph Tokens

```text
Paragraph
    ↓
Count tokens
    ↓
Is token_count <= 512?
```

### Case A — Paragraph fits

Example:

```text
Paragraph = 300 tokens
```

Since:

```text
300 <= 512
```

the complete paragraph becomes one `SemanticUnit`.

```python
SemanticUnit(
    text=paragraph,
    token_count=300,
)
```

Flow:

```text
Paragraph
    ↓
300 tokens
    ↓
SemanticUnit
```

---

### Case B — Paragraph is too large

Example:

```text
Paragraph = 900 tokens
```

Since:

```text
900 > 512
```

the paragraph cannot become one semantic unit.

Therefore:

```text
Paragraph
    ↓
Split into sentences
```

---

# 6. Split Oversized Paragraph into Sentences

Example:

```text
900-token paragraph
        ↓
Sentence 1 → 200 tokens
Sentence 2 → 150 tokens
Sentence 3 → 700 tokens
Sentence 4 → 100 tokens
```

Now process each sentence separately.

---

# 7. Take One Sentence

For each sentence:

```text
Sentence
    ↓
Count tokens
```

### If sentence <= 512 tokens

Example:

```text
Sentence = 200 tokens
```

Since:

```text
200 <= 512
```

create:

```python
SemanticUnit(
    text=sentence,
    token_count=200,
)
```

---

### If sentence > 512 tokens

Example:

```text
Sentence = 700 tokens
```

Since:

```text
700 > 512
```

the sentence is too large.

Therefore:

```text
Sentence
    ↓
Split by words
```

---

# 8. Split Oversized Sentence by Words

The sentence is split into words.

```text
Sentence
    ↓
word1
word2
word3
word4
...
```

The chunker maintains a word bucket:

```text
current_words = []
current_tokens = 0
```

It takes one word at a time.

For every word:

```text
Take word
    ↓
Count word tokens
    ↓
Check if word can fit in current bucket
```

---

# 9. Add Words to the Bucket

Example:

```text
Current bucket = 450 tokens
Next word = 40 tokens
```

The projected size is:

```text
450 + 40 = 490
```

Since:

```text
490 <= 512
```

the word is added.

```text
Bucket
├── words...
└── total = 490 tokens
```

Continue taking words.

---

# 10. When Adding the Next Word Exceeds 512

Example:

```text
Current bucket = 490 tokens
Next word = 40 tokens
```

Projected:

```text
490 + 40 = 530
```

Since:

```text
530 > 512
```

the new word is not added to the current bucket.

The current bucket is flushed and becomes a `SemanticUnit`.

```python
SemanticUnit(
    text="words in current bucket",
    token_count=actual_tokens,
)
```

Then a new bucket starts with the word that could not fit.

```text
Old bucket
    ↓
SemanticUnit

New bucket
    ↓
Continue adding words
```

---

# 11. Special Case — One Word Itself Is Larger Than 512 Tokens

If one individual word is larger than the maximum:

```text
word = 700 tokens
MAX_TOKENS = 512
```

It cannot fit into any normal word bucket.

Therefore:

```text
700-token word
    ↓
Split by characters
```

The characters are then accumulated into token-bounded pieces.

```text
Characters
    ↓
Build character bucket
    ↓
<= 512 tokens
    ↓
SemanticUnit
```

So even an extremely long word is prevented from creating an oversized semantic unit.

---

# 12. Result of the Paragraph Processing

After processing all paragraphs:

```text
Paragraphs
    ↓
Sentences / words / characters when necessary
    ↓
SemanticUnits
```

Every resulting unit has:

```python
SemanticUnit(
    text=...,
    token_count=...,
)
```

Example:

```text
Semantic Units

Unit 1 → 212 tokens
Unit 2 → 215 tokens
Unit 3 → 510 tokens
Unit 4 → 144 tokens
```

---

# 13. Process Every Section

This process happens for every section of the patent.

```text
Patent
│
├── Section 1
│   ├── Paragraphs
│   └── Semantic Units
│
├── Section 2
│   ├── Paragraphs
│   └── Semantic Units
│
└── Section 3
    ├── Paragraphs
    └── Semantic Units
```

The semantic units are then passed to the chunk-building stage.

---

# 14. Build Patent Chunks from Semantic Units

Now the chunk builder takes the semantic units **section-wise** and tries to combine consecutive units while staying within the maximum token limit.

Assume:

```text
MAX_TOKENS = 512
```

Section 1 has:

```text
Unit 1 = 212 tokens
Unit 2 = 215 tokens
Unit 3 = 510 tokens
Unit 4 = 144 tokens
```

---

# 15. Take Unit 1

```text
Unit 1 = 212
```

Start a chunk:

```text
Chunk 1
└── Unit 1 = 212
```

Current total:

```text
212 tokens
```

---

# 16. Add Unit 2

```text
Unit 2 = 215
```

Check:

```text
212 + 215 = 427
```

Since:

```text
427 <= 512
```

Unit 2 is added to the same chunk.

```text
Chunk 1
├── Unit 1 = 212
└── Unit 2 = 215

Total = 427 tokens
```

---

# 17. Try to Add Unit 3

```text
Unit 3 = 510
```

Check:

```text
427 + 510 = 937
```

Since:

```text
937 > 512
```

Unit 3 cannot fit.

Therefore, the current chunk is finalized:

```text
Chunk 1
├── Unit 1
└── Unit 2

Total = 427 tokens
```

---

# 18. Unit 3 Starts a New Chunk

Now Unit 3 starts its own chunk:

```text
Chunk 2
└── Unit 3 = 510 tokens
```

Current total:

```text
510 tokens
```

---

# 19. Try to Add Unit 4

```text
Unit 4 = 144 tokens
```

Check:

```text
510 + 144 = 654
```

Since:

```text
654 > 512
```

Unit 4 cannot fit.

Therefore:

```text
Chunk 2
└── Unit 3

Total = 510 tokens
```

is finalized.

Then Unit 4 becomes:

```text
Chunk 3
└── Unit 4 = 144 tokens
```

---

# 20. Final Result for This Section

The original 4 semantic units become **3 patent chunks**:

```text
Semantic Units:

Unit 1 = 212
Unit 2 = 215
Unit 3 = 510
Unit 4 = 144


            ↓


Patent Chunks:

Chunk 1
├── Unit 1 = 212
└── Unit 2 = 215
    Total = 427

Chunk 2
└── Unit 3 = 510
    Total = 510

Chunk 3
└── Unit 4 = 144
    Total = 144
```

---

# 21. The Important Hierarchy

The complete hierarchy is:

```text
PATENT
  │
  ├── SECTION
  │     │
  │     ├── PARAGRAPH
  │     │      │
  │     │      ├── SENTENCE
  │     │      │
  │     │      ├── WORD
  │     │      │
  │     │      └── CHARACTER
  │     │
  │     └── SEMANTIC UNITS
  │
  └── PATENT CHUNKS
```

But the lower-level splitting only happens when necessary:

```text
Paragraph
    │
    ├── fits 512 tokens
    │      ↓
    │   SemanticUnit
    │
    └── exceeds 512
           ↓
        Sentences
           │
           ├── sentence fits
           │      ↓
           │   SemanticUnit
           │
           └── sentence exceeds
                  ↓
                Words
                  │
                  ├── words fit
                  │      ↓
                  │   SemanticUnit
                  │
                  └── individual word exceeds
                         ↓
                      Characters
                         ↓
                    SemanticUnit
```

---

# 22. Final Chunking Flow

```text
                    PATENT_DIRECTORY
                           │
                           ▼
                      Patent File
                           │
                           ▼
                     PatentParser
                           │
                           ▼
                   Detect Sections
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Section 1        Section 2        Section 3
          │                │                │
          ▼                ▼                ▼
      Paragraphs        Paragraphs        Paragraphs
          │                │                │
          ▼                ▼                ▼
     Process each      Process each      Process each
      paragraph         paragraph         paragraph
          │                │                │
          ▼                ▼                ▼
     Check size        Check size        Check size
          │
          ├── Fits 512 tokens
          │       ↓
          │   SemanticUnit
          │
          └── Too large
                  ↓
              Sentences
                  │
                  ├── Fits 512
                  │      ↓
                  │  SemanticUnit
                  │
                  └── Too large
                         ↓
                       Words
                         │
                         ├── Fit
                         │    ↓
                         │ SemanticUnit
                         │
                         └── Word > 512
                                ↓
                            Characters
                                ↓
                           SemanticUnit
                                  │
                                  ▼
                         Section Semantic Units
                                  │
                                  ▼
                         Build chunks section-wise
                                  │
                                  ▼
                           Patent Chunks
                                  │
                                  ▼
                              return chunks
```

---

# 23. After Chunking — Embedding and Insertion Flow

Once the `PatentChunker` finishes processing one patent, we have:

```text
Patent
│
├── Metadata
│
└── Patent Chunks
    ├── Chunk 1
    ├── Chunk 2
    ├── Chunk 3
    └── ...
```

The chunking process returns:

```text
document + chunks
```

These are then passed to the **prefetch queue**.

---

# 24. Put the Processed Patent into the Prefetch Queue

The producer puts the complete result into:

```python
queue.put(result)
```

The queue item contains:

```text
(
    txt_file,
    document,
    chunks,
    error
)
```

So one queue item represents:

```text
ONE PATENT
│
├── Patent file
├── Parsed document
├── ALL chunks of that patent
└── Error information
```

Example:

```text
P1
│
├── document
│
└── chunks
    ├── Chunk 1
    ├── Chunk 2
    ├── Chunk 3
    └── Chunk 4
```

This complete P1 result is placed into the prefetch queue.

---

# 25. Main Thread Takes One Patent from the Prefetch Queue

The main ingestion loop waits on:

```python
item = queue.get()
```

This removes one complete patent result from the prefetch queue.

Example:

```text
PREFETCH QUEUE

┌─────────────────────┐
│ P1 → 4 chunks       │
├─────────────────────┤
│ P2 → 7 chunks       │
├─────────────────────┤
│ P3 → 5 chunks       │
└─────────────────────┘
```

The main thread executes:

```python
item = queue.get()
```

and takes P1.

Now:

```text
PREFETCH QUEUE

P2 → 7 chunks
P3 → 5 chunks
```

The main thread has:

```text
P1
│
├── document
└── 4 chunks
```

---

# 26. Separate Patent Metadata

The main thread takes the metadata from the document and adds it to:

```text
metadata_batch
```

Example:

```text
metadata_batch

P1
├── patent_id
├── title
├── assignee
├── application information
└── other metadata
```

This metadata is temporarily held.

It is **not written to Qdrant immediately**.

---

# 27. Add the Patent's Chunks to `embed_pool`

The chunks from P1 are added to:

```python
embed_pool.extend(chunks)
```

Example:

```text
P1 = 4 chunks

embed_pool

├── P1 Chunk 1
├── P1 Chunk 2
├── P1 Chunk 3
└── P1 Chunk 4
```

The code also keeps track of which patent contributed those chunks.

This is stored in:

```text
pool_patents
```

Example:

```text
pool_patents

P1 → 4 chunks → P1.txt
```

This association is important later so the system knows which patent was represented by the embedded chunks.

---

# 28. Continue Taking More Patents

The main thread does **not necessarily embed P1 immediately**.

It checks whether enough chunks have accumulated for an embedding batch.

For example:

```text
EMBED_BATCH_SIZE = 256
```

Suppose:

```text
P1 → 80 chunks
```

Then:

```text
embed_pool = 80 chunks
```

Check:

```text
80 >= 256
```

FALSE.

So the main thread takes another patent:

```text
P2 → 100 chunks
```

Now:

```text
embed_pool

P1 → 80
P2 → 100
-----------
Total = 180 chunks
```

Still:

```text
180 >= 256
```

FALSE.

So it takes another patent.

---

# 29. Continue Until Embedding Threshold Is Reached

Suppose P3 has 100 chunks.

Now:

```text
embed_pool

P1 → 80
P2 → 100
P3 → 100
-----------
Total = 280 chunks
```

Now:

```text
280 >= 256
```

TRUE.

Therefore the embedding stage starts.

---

# 30. Send the Chunks to the Embedding Model

The code calls:

```python
embedder.embed_batch(embed_pool, ...)
```

So:

```text
280 chunks
     ↓
Embedding Model
     ↓
280 embedding vectors
```

Conceptually:

```text
P1 Chunk 1
    ↓
Embedding Model
    ↓
Vector 1

P1 Chunk 2
    ↓
Embedding Model
    ↓
Vector 2

P2 Chunk 1
    ↓
Embedding Model
    ↓
Vector 3

...
```

The important transformation is:

```text
TEXT CHUNK
    ↓
EMBEDDING MODEL
    ↓
VECTOR
```

For example:

```text
Chunk:
"The display panel includes..."

        ↓

Embedding Model

        ↓

[0.012, -0.238, 0.551, ...]
```

The vector has the embedding model's configured dimensionality.

---

# 31. Move the Embedded Chunks into `batch`

After embedding, the embedded chunks are added to:

```text
batch
```

Conceptually:

```text
embed_pool
    ↓
Embedding
    ↓
Embedded chunks
    ↓
batch
```

Example:

```text
batch

├── P1 Chunk 1 + Vector
├── P1 Chunk 2 + Vector
├── ...
├── P2 Chunk 1 + Vector
└── ...
```

At the same time, the patent information is moved into:

```text
pending_patents
```

Example:

```text
pending_patents

P1 → 80 chunks
P2 → 100 chunks
P3 → 100 chunks
```

This tells the writer:

> These patents are represented by the embedded data currently waiting to be written.

---

# 32. Clear `embed_pool`

After moving the embedded data into `batch`, the temporary embedding pool is cleared:

```text
embed_pool = []
pool_patents = []
```

Now it is ready to collect chunks from the next patents.

So:

```text
OLD

embed_pool
└── 280 chunks

        ↓ embedding

batch
└── 280 embedded chunks


NEW

embed_pool
└── EMPTY
```

---

# 33. Check Whether the Write Batch Is Large Enough

Now the system checks:

```text
Is the write batch large enough?
```

For example:

```text
BATCH_SIZE = 256
```

If:

```text
len(batch) >= BATCH_SIZE
```

then the data is ready to be handed to the writer.

There is also a metadata batch threshold:

```text
len(metadata_batch) >= METADATA_BATCH_SIZE
```

So a write can be triggered when either the embedded batch or metadata batch reaches its configured threshold.

---

# 34. Put the Embedded Batch into the Write Queue

When the write batch is ready:

```python
write_queue.put(
    (
        metadata_batch,
        batch,
        pending_patents,
        False
    )
)
```

Now the data moves into the **write queue**.

The write queue item contains:

```text
Write Job
│
├── metadata_batch
├── embedded chunk batch
├── pending_patents
└── flush/control information
```

Example:

```text
WRITE QUEUE

┌──────────────────────────────┐
│ Write Batch #1               │
│                              │
│ Metadata: P1, P2, P3         │
│                              │
│ Embedded chunks:             │
│ P1-C1 → Vector               │
│ P1-C2 → Vector               │
│ P2-C1 → Vector               │
│ ...                          │
│                              │
│ Pending patents: P1, P2, P3  │
└──────────────────────────────┘
```

---

# 35. `_WRITE_QUEUE_DEPTH = 2`

If:

```python
write_queue = Queue(maxsize=2)
```

then at most **2 write jobs can wait in the write queue**.

Example:

```text
WRITE QUEUE

┌───────────────────────┐
│ Write Batch #1        │
└───────────────────────┘

┌───────────────────────┐
│ Write Batch #2        │
└───────────────────────┘
```

If the queue is full and the main thread tries:

```python
write_queue.put(...)
```

the main thread waits until the writer removes a job.

This creates **backpressure**.

---

# 36. Writer Thread Takes the Write Job

Separately, a writer thread is running.

Conceptually:

```python
job = write_queue.get()
```

It takes:

```text
Write Batch #1
```

from the write queue.

Now:

```text
WRITE QUEUE

Write Batch #2
```

The writer now processes Batch #1.

---

# 37. Writer Calls the Flush/Write Operation

The writer passes the job to the database-writing logic.

Conceptually:

```text
Write Batch
     ↓
flush()
     ↓
Qdrant database
```

The writer has the information needed to insert:

```text
Patent metadata
+
Chunk text
+
Chunk vectors
+
Chunk/patent relationships
```

---

# 38. Insert Patent-Level Metadata

Patent-level information is stored in the patent collection.

Conceptually:

```text
Patent metadata
      ↓
PATENTS COLLECTION
```

Example:

```text
P1
├── patent_id
├── title
├── assignee
├── country
├── application year
└── other metadata
```

So the patent can later be retrieved using its metadata.

---

# 39. Insert Chunk-Level Vectors

The chunk data is stored in the chunk collection.

Conceptually:

```text
Chunk text + embedding vector + metadata
                ↓
       PATENT CHUNKS COLLECTION
```

Example:

```text
P1 Chunk 1
├── chunk text
├── vector
├── patent_id
├── section information
└── other chunk metadata
```

Then:

```text
P1 Chunk 2
├── chunk text
├── vector
└── metadata
```

and so on.

So Qdrant contains the searchable vector representation of the patent chunks.

---

# 40. Qdrant Insertion

The final database flow is:

```text
                  WRITE BATCH
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Patent Metadata          Chunk + Vector
          │                       │
          ▼                       ▼
 PATENTS COLLECTION       PATENT_CHUNKS COLLECTION
          │                       │
          └───────────┬───────────┘
                      ▼
                   QDRANT
```

At this point the patent's data has been indexed.

---

# 41. Mark the Patent as Successfully Written

After the writer successfully completes the database operation, the processed patents are acknowledged/recorded.

The important concept is:

```text
Patent
   ↓
Parsed
   ↓
Chunked
   ↓
Embedded
   ↓
Written to Qdrant successfully
   ↓
Mark as processed
```

The checkpoint/progress mechanism prevents the patent from unnecessarily being processed again on a future indexing run.

---

# 42. Continue With the Next Patents

While the writer is writing Batch #1, the producer can continue parsing and chunking more patents, and the main thread can continue collecting and embedding more chunks, subject to the queue limits.

So the system is **pipeline-based**:

```text
                    ┌──────────────────────┐
                    │   PRODUCER WORKERS   │
                    │                      │
Patent files ──────►│ Parse + Chunk        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   PREFETCH QUEUE     │
                    │                      │
                    │ Parsed + Chunked     │
                    │ Patents               │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     MAIN THREAD      │
                    │                      │
                    │ Take patents         │
                    │ Collect chunks       │
                    │ Build embed_pool     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      EMBEDDING       │
                    │                      │
                    │ Chunks → Vectors     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       BATCH          │
                    │                      │
                    │ Embedded chunks      │
                    │ + metadata           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     WRITE QUEUE      │
                    │      maxsize=2       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    WRITER THREAD     │
                    │                      │
                    │ Flush / Insert       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       QDRANT         │
                    │                      │
                    │ Patent collection    │
                    │ Chunk collection     │
                    └──────────┬───────────┘
                               │
                               ▼
                         CHECKPOINT
```

# 43. Complete End-to-End Flow

The complete indexing flow is therefore:

```text
PATENT_DIRECTORY
      │
      ▼
Take Patent File
      │
      ▼
PATENT PARSER
      │
      ▼
Detect Sections
      │
      ▼
Section-wise Content
      │
      ▼
Split into Paragraphs
      │
      ▼
Process Paragraph
      │
      ├── Paragraph <= 512 tokens
      │       ↓
      │   SemanticUnit
      │
      └── Paragraph > 512
              ↓
          Split Sentences
              │
              ├── Sentence <= 512
              │       ↓
              │   SemanticUnit
              │
              └── Sentence > 512
                      ↓
                  Split Words
                      │
                      ├── Words fit
                      │      ↓
                      │  SemanticUnit
                      │
                      └── Word > 512
                             ↓
                         Split Characters
                             ↓
                         SemanticUnit
                              │
                              ▼
                    Build Section-wise Chunks
                              │
                              ▼
                       PATENT CHUNKS
                              │
                              ▼
                       PREFETCH QUEUE
                              │
                              ▼
                     Main Thread `queue.get()`
                              │
                              ▼
                    Add Metadata → metadata_batch
                              │
                              ▼
                    Add Chunks → embed_pool
                              │
                              ▼
                    Accumulate chunks
                              │
                              ▼
                    Embedding threshold reached
                              │
                              ▼
                       EMBEDDING MODEL
                              │
                              ▼
                       CHUNKS → VECTORS
                              │
                              ▼
                            batch
                              │
                              ▼
                    Check write batch size
                              │
                              ▼
                       WRITE QUEUE
                              │
                              ▼
                       WRITER THREAD
                              │
                              ▼
                           flush()
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          Patent Metadata          Chunk + Vector
                  │                       │
                  ▼                       ▼
        PATENTS COLLECTION       PATENT_CHUNKS COLLECTION
                  │                       │
                  └───────────┬───────────┘
                              ▼
                           QDRANT
                              │
                              ▼
                         CHECKPOINT
                              │
                              ▼
                       NEXT PATENT
```

# 44. Simple One-Line Summary

```text
Patent File
→ Parse
→ Detect Sections
→ Paragraphs
→ Semantic Units
→ Section-wise Patent Chunks
→ Prefetch Queue
→ Main Thread
→ Collect Chunks
→ Embedding
→ Vectors
→ Write Batch
→ Write Queue
→ Writer
→ Qdrant
→ Checkpoint
→ Next Patent
```
