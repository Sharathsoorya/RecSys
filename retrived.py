import torch
import pprint
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

COLLECTION_NAME = "multimodal_collection_new"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def chunk_text(text, max_tokens=70):
    tokenizer = clip_processor.tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def get_query_embedding_advanced(image_path: str = None, text: str = None, text_weight=0.5, image_weight=0.5):
    text_embedding = None
    image_embedding = None

    if text:
        text_chunks = chunk_text(text)
        text_chunk_embeddings = []

        for chunk in text_chunks:
            inputs = clip_processor(
                text=chunk,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            with torch.no_grad():
                text_emb = clip_model.get_text_features(
                    inputs['input_ids'], inputs['attention_mask']
                )
                text_chunk_embeddings.append(text_emb)

        text_embedding = torch.mean(torch.stack(text_chunk_embeddings), dim=0)

    if image_path:
        image_chunk_embeddings = []

        for _ in range(3):
            inputs = clip_processor(
                images=Image.open(image_path).convert("RGB"),
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            with torch.no_grad():
                img_emb = clip_model.get_image_features(inputs['pixel_values'])
                image_chunk_embeddings.append(img_emb)

        image_embedding = torch.mean(torch.stack(image_chunk_embeddings), dim=0)

    if text_embedding is not None and image_embedding is not None:
        combined_emb = torch.cat([
            text_embedding * text_weight,
            image_embedding * image_weight
        ], dim=-1)
    elif text_embedding is not None:
        combined_emb = torch.cat([text_embedding, text_embedding], dim=-1)
    elif image_embedding is not None:
        combined_emb = torch.cat([image_embedding, image_embedding], dim=-1)
    else:
        raise ValueError("Either text or image_path must be provided")

    return combined_emb.cpu().numpy()[0]

def search_fashion_query(image_path=None, text=None, top_k=5):
    if not image_path and not text:
        raise ValueError("Either image_path or text must be provided")

    query_vector = get_query_embedding_advanced(image_path=image_path, text=text)
    query_type = "text only" if text and not image_path else "image only" if image_path and not text else "multimodal"
    print(f"Performing {query_type} search...")

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

    return results

def get_retrieved_items(image_path=None, text=None, top_k=5):
    """Return Qdrant search results as a list of payload dictionaries with all metadata."""
    results = search_fashion_query(image_path=image_path, text=text, top_k=top_k)
    items = []
    print("------------------Before Reranking------------------")
    for res in results:
        p = res.payload
        image_path_in_db = p.get("image_path", "")
        if image_path_in_db:
            path_parts = image_path_in_db.split('/')
            repo_name = '/'.join(path_parts[:2])
            image_file = path_parts[-1]
            hf_image_url = f"https://huggingface.co/datasets/{repo_name}/resolve/main/{image_file}"
            print(f"   Image: {hf_image_url}")
        else:
            print(f"   Image: Not available")

        items.append({
            "id": res.id,
            "score": res.score,
            "gender": p.get("gender_x", "N/A"),
            "masterCategory": p.get("masterCategory_x", "N/A"),
            "subCategory": p.get("subCategory_x", "N/A"),
            "articleType": p.get("articleType_x", "N/A"),
            "baseColour": p.get("baseColour_x", "N/A"),
            "season": p.get("season_x", "N/A"),
            "year": p.get("year_x", "N/A"),
            "usage": p.get("usage_x", "N/A"),
            "product": p.get("productDisplayName_x", "N/A"),
            "brandName": p.get("brandName", "N/A"),
            "articleAttributes": p.get("articleAttributes", "N/A"),
            "description": p.get("description", "N/A"),
            "price": p.get("price", "N/A"),
            "rating": p.get("rating", "N/A"),
            "image_path": hf_image_url if image_path_in_db else "Not available"
        })
    return items


