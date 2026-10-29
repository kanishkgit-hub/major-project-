import chromadb
from collections import Counter

from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


def main():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    collection = client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )

    print("=" * 60)
    print("CHROMADB METADATA DIAGNOSTICS")
    print("=" * 60)

    print("Database path:", CHROMA_DB_DIR)
    print("Collection:", collection.name)
    print("Total chunks:", collection.count())

    # Fetch all metadata to inspect actual stored values.
    data = collection.get(
        limit=collection.count(),
        include=["metadatas"],
    )

    metadata_list = data["metadatas"]

    print("\nTOTAL METADATA RECORDS:", len(metadata_list))

    # Print exact values, types, and representations.
    categories = Counter()

    for metadata in metadata_list:
        if metadata:
            value = metadata.get("disease")
            categories[repr(value)] += 1

    print("\nEXACT DISEASE VALUES:")
    for value, count in categories.items():
        print(f"{value}: {count}")

    print("\nFILTER TESTS:")

    for disease in ["diabetes", "heart", "kidney"]:
        print(f"\nTesting category: {disease!r}")

        # Test the filter directly.
        try:
            result = collection.get(
                where={"disease": disease},
                limit=10,
                include=["metadatas"],
            )

            print(
                "Direct filter matches:",
                len(result["ids"]),
            )

            print(
                "Returned metadata:",
                result["metadatas"][:3],
            )

        except Exception as error:
            print("Direct filter error:", repr(error))

        # Independently count matching metadata.
        manual_matches = [
            metadata
            for metadata in metadata_list
            if metadata
            and metadata.get("disease") == disease
        ]

        print(
            "Manual exact matches:",
            len(manual_matches),
        )

    print("\nDIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
