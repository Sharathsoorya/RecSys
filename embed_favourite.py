from qdrant_client import QdrantClient, models

# --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MAIN_COLLECTION = "multimodal_collection_new"  # where embeddings are stored

# --- Initialize Client ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def find_item_by_url(image_url: str):
    """
    Given an image URL, find matching item in Qdrant collection
    using the 'image_path' payload field.
    Prints vector length, payload keys, and vector values.
    Returns both vector and payload.
    """
    try:
        # Extract filename (e.g., "13089.jpg")
        image_filename = image_url.split("/")[-1]
        print(f"🔍 Searching for image: {image_filename}")

        # Create filter for match
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="image_path",
                    match=models.MatchValue(
                        value=f"Sharathsoorya/Multimodal_Fashion_Data/{image_filename}"
                    )
                )
            ]
        )

        # Scroll to find match
        scroll_results, _ = client.scroll(
            collection_name=MAIN_COLLECTION,
            scroll_filter=search_filter,
            limit=1,
            with_payload=True,
            with_vectors=True
        )

        if not scroll_results:
            print("⚠️ No matching image found in Qdrant.")
            return None

        # First matched point
        point = scroll_results[0]

        # Extract vector + payload
        vector = point.vector
        payload = point.payload

        print("\n✅ Found item in Qdrant:")
        print(f"🧠 Vector length: {len(vector)}D")
        print(f"🖼️ Image URL: {image_url}")
        print(f"📦 Payload keys: {list(payload.keys())}\n")

        # Print only first few vector values to avoid flooding console
        preview = vector[:10]  # show first 10 dims
        print(f"🔢 Vector preview (first 10 dims): {preview}\n")

        return {
            "vector": vector,
            "payload": payload,
            "image_url": image_url
        }

    except Exception as e:
        print(f"❌ Error finding item: {e}")
        return None
