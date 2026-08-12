from pathlib import Path

from app.parser import PatentParser


def main():

    parser = PatentParser()

    patent = parser.load_patent(
        Path("patents-processed/AP170S1.txt")
    )

    print("=" * 60)

    print("Patent ID:")
    print(patent.patent_id)

    print("\nText:\n")
    print(patent.text)

    print("\nMetadata:\n")

    for key, value in patent.metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
