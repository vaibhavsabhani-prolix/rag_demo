from app.qdrant_db import QdrantDB


def main():

    db = QdrantDB()

    db.create_collections()

    print()

    info = db.get_collection_info()

    print("Collection Name :", info.config.params.vectors)

    print()

    print("Total Points :", db.count_points())


if __name__ == "__main__":
    main()