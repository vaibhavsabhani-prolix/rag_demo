"""
Qdrant Indexed Patents Overview

Scrolls Qdrant vector collection, groups chunks by patent_id,
aggregates statistics, and prints a formatted summary table.
"""

from collections import defaultdict
import sys
from app.qdrant_db import QdrantDB
from app.config import COLLECTION_NAME


def fetch_all_chunks(db: QdrantDB):
    """
    Scroll through Qdrant collection to retrieve all indexed point payloads.
    """
    points = []
    next_page_offset = None

    while True:
        records, next_page_offset = db.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(records)
        if not next_page_offset or not records:
            break

    return points


def print_table(patents_data: list[dict]):
    """
    Render a clean ASCII table of indexed patents.
    """
    headers = [
        "#",
        "Patent ID",
        "Chunks",
        "Sections",
        "Tokens",
        "Words",
        "Assignee / Inventor",
        "Pub. Year",
    ]

    # Calculate column widths dynamically
    col_widths = [len(h) for h in headers]

    rows = []
    for idx, p in enumerate(patents_data, start=1):
        assignee = p["assignee"]
        if len(assignee) > 25:
            assignee = assignee[:22] + "..."

        row = [
            str(idx),
            p["patent_id"],
            str(p["chunk_count"]),
            str(len(p["sections"])),
            f"{p['total_tokens']:,}",
            f"{p['total_words']:,}",
            assignee,
            str(p["pub_year"]),
        ]
        rows.append(row)
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Format string generator
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    divider = "-+-".join(["-" * w for w in col_widths])

    print("+" + "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2) + "+")
    print("| " + row_fmt.format(*headers) + " |")
    print("+" + divider.replace("-+-", "-+-") + "+")
    for r in rows:
        print("| " + row_fmt.format(*r) + " |")
    print("+" + "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2) + "+")


def main():
    print("\nConnecting to Qdrant Database...")
    db = QdrantDB()

    total_vectors = db.count_points()
    print(f"Collection '{COLLECTION_NAME}' contains {total_vectors:,} total vectors.\n")

    if total_vectors == 0:
        print("No vectors found in Qdrant collection. Run 'python -m app.ingest' first.")
        return

    print("Fetching points from Qdrant...")
    points = fetch_all_chunks(db)

    # Aggregate by patent_id
    patents = defaultdict(lambda: {
        "chunks": [],
        "sections": set(),
        "total_tokens": 0,
        "total_words": 0,
        "metadata": {},
    })

    for pt in points:
        payload = pt.payload
        pid = payload.get("patent_id", "UNKNOWN")
        patents[pid]["chunks"].append(payload)
        sec = payload.get("section", "UNKNOWN")
        if sec:
            patents[pid]["sections"].add(sec)
        patents[pid]["total_tokens"] += payload.get("token_count", 0)
        patents[pid]["total_words"] += payload.get("word_count", 0)

        if not patents[pid]["metadata"] and "metadata" in payload:
            patents[pid]["metadata"] = payload.get("metadata", {})

    patent_list = []
    for pid, data in patents.items():
        meta = data["metadata"]

        # Extract assignee or inventor fallback
        assignee_arr = meta.get("Original Assignee First") or meta.get("Original Assignee") or meta.get("Assignee Standardized")
        if isinstance(assignee_arr, list) and assignee_arr:
            assignee = assignee_arr[0]
        elif isinstance(assignee_arr, str):
            assignee = assignee_arr
        else:
            inventor_arr = meta.get("Inventor First") or meta.get("Inventor")
            if isinstance(inventor_arr, list) and inventor_arr:
                assignee = f"Inv: {inventor_arr[0]}"
            elif isinstance(inventor_arr, str):
                assignee = f"Inv: {inventor_arr}"
            else:
                assignee = "N/A"

        pub_year = meta.get("Publication Year") or meta.get("Application Year") or "N/A"

        patent_list.append({
            "patent_id": pid,
            "chunk_count": len(data["chunks"]),
            "sections": sorted(list(data["sections"])),
            "total_tokens": data["total_tokens"],
            "total_words": data["total_words"],
            "assignee": assignee,
            "pub_year": pub_year,
            "raw_chunks": data["chunks"],
        })

    # Sort patents by chunk count descending
    patent_list.sort(key=lambda x: x["chunk_count"], reverse=True)

    print("\n" + "=" * 80)
    print("                      INDEXED PATENTS SUMMARY (QDRANT)")
    print("=" * 80)
    print(f" Total Patents Indexed : {len(patent_list)}")
    print(f" Total Chunks/Vectors  : {len(points):,}")
    print(f" Total Indexed Tokens  : {sum(p['total_tokens'] for p in patent_list):,}")
    print(f" Total Indexed Words   : {sum(p['total_words'] for p in patent_list):,}")
    print(f" Avg Chunks per Patent : {len(points) / len(patent_list):.1f}")
    print("=" * 80 + "\n")

    print_table(patent_list)

    # Check if user wants detailed breakdown
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print("\n" + "=" * 80)
        print("                        DETAILED SECTION BREAKDOWN")
        print("=" * 80)
        for p in patent_list:
            print(f"\nPatent ID : {p['patent_id']}")
            print(f"Assignee  : {p['assignee']} | Pub Year: {p['pub_year']}")
            print(f"Chunks    : {p['chunk_count']} | Tokens: {p['total_tokens']:,} | Words: {p['total_words']:,}")
            print("Sections  :")
            section_counts = defaultdict(int)
            for c in p["raw_chunks"]:
                section_counts[c.get("section", "UNKNOWN")] += 1
            for sec, cnt in sorted(section_counts.items()):
                print(f"  - {sec:35s} : {cnt} chunk(s)")


if __name__ == "__main__":
    main()
