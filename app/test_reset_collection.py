from app.qdrant_db import QdrantDB


def main():

    db = QdrantDB()

    print("\nBefore Reset")

    print("----------------")

    print("Points :", db.count_points())

    print()

    db.reset_collections()

    print()

    print("After Reset")

    print("----------------")

    print("Points :", db.count_points())


if __name__ == "__main__":
    main()