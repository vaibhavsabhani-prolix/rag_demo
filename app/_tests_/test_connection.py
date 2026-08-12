from app.qdrant_db import QdrantDB


def main():

    db = QdrantDB()

    print("\nConnected to Qdrant successfully!\n")

    db.create_collections()

    print("\nAvailable Collections:\n")

    collections = db.client.get_collections()

    for collection in collections.collections:
        print(f"- {collection.name}")


if __name__ == "__main__":
    main()
