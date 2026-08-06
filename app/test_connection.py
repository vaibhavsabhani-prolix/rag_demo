from app.qdrant_db import (
    get_qdrant_client,
    create_collection,
)


def main():

    client = get_qdrant_client()

    print("\nConnected to Qdrant successfully!\n")

    create_collection()

    print("\nAvailable Collections:\n")

    collections = client.get_collections()

    for collection in collections.collections:
        print(f"- {collection.name}")


if __name__ == "__main__":
    main()
