import os
import json
import requests
import uuid
from typing import List, Dict, Tuple, Optional
from textwrap import dedent
from dotenv import load_dotenv
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Set it in your environment variables or .env file.")

LLM_MODEL = "meta-llama/llama-3-8b-instruct"

client = None

PRODUCT_COLLECTION = "multimodal_collection_new"
REG_COLLECTION = "registration_collection"
FAVOURITES_COLLECTION = "favourites_collection"

def set_qdrant_client(qdrant_client: QdrantClient):
    """Set the Qdrant client from main app"""
    global client
    client = qdrant_client
    print(f"Qdrant client set in rerank_llm")

def safe_scroll(collection_name, scroll_filter, limit=100, with_vectors=True):
    """Handle Qdrant scroll unpacking variations safely."""
    if client is None:
        print("Qdrant client not initialized in rerank_llm")
        return []
    
    try:
        res = client.scroll(
            collection_name=collection_name, 
            scroll_filter=scroll_filter, 
            limit=limit,
            with_vectors=with_vectors
        )
        if isinstance(res, tuple):
            return res[0] or []
        return res or []
    except Exception as e:
        print(f"Scroll failed for {collection_name}: {e}")
        return []
    
def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "FashionRAG-Reranker",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.25,
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def get_user_point_by_email(email: str) -> Optional[Dict]:
    """Return the registration point dict for given email, or None."""
    if client is None or not client.collection_exists(REG_COLLECTION):
        return None
    filt = Filter(must=[FieldCondition(key="email", match=MatchValue(value=email))])
    pts = safe_scroll(REG_COLLECTION, filt, limit=1)
    return pts[0] if pts else None

def increment_login_count(email: str):
    """Increment login count for user stored in registration collection."""
    if client is None:
        return
        
    user = get_user_point_by_email(email)
    if not user:
        return
    user_id = user.id if hasattr(user, "id") else user.get("id", str(uuid.uuid4()))
    payload = user.payload.copy() if hasattr(user, "payload") else user.get("payload", {}).copy()
    payload["login_count"] = int(payload.get("login_count", 0)) + 1
    payload["last_login"] = datetime.utcnow().isoformat()
    client.upsert(collection_name=REG_COLLECTION, points=[{"id": user_id, "payload": payload}])

def get_login_count(email: str) -> int:
    u = get_user_point_by_email(email)
    if not u:
        return 0
    payload = u.payload if hasattr(u, "payload") else u.get("payload", {})
    return int(payload.get("login_count", 0))

def get_most_recent_favourite_vector(email: str) -> Optional[List[float]]:
    """Get the most recent favorite vector for a user"""
    if client is None:
        print("Qdrant client not initialized")
        return None
    
    print(f"\n===== GETTING MOST RECENT FAVORITE FOR: {email} =====")
    
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"Available collections: {collection_names}")
        
        if FAVOURITES_COLLECTION not in collection_names:
            print(f"Collection '{FAVOURITES_COLLECTION}' not found!")
            return None
        
        filt = Filter(must=[
            FieldCondition(key="email", match=MatchValue(value=email))
        ])
        
        print(f"Searching for favorites with email: {email}")

        points = safe_scroll(FAVOURITES_COLLECTION, filt, limit=50, with_vectors=True)
        print(f"Found {len(points)} favorite points for {email}")
        
        if not points:
            print(f"No favorites found for email: {email}")
            return None
        
        most_recent_point = None
        most_recent_time = None
        
        for p in points:
            payload = p.payload if hasattr(p, "payload") else p.get("payload", {})
            timestamp_str = payload.get("added_on", "")

            try:
                if timestamp_str:
                    if "T" in timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    if most_recent_time is None or timestamp > most_recent_time:
                        most_recent_time = timestamp
                        most_recent_point = p
            except Exception as e:
                print(f"Failed to parse timestamp '{timestamp_str}': {e}")
                if most_recent_point is None:
                    most_recent_point = p
        
        if most_recent_point is None:
            print(f"Could not determine most recent favorite")
            return None
        
        payload = most_recent_point.payload if hasattr(most_recent_point, "payload") else most_recent_point.get("payload", {})
        point_id = getattr(most_recent_point, "id", "unknown")
        
        vector = getattr(most_recent_point, "vector", None) or (most_recent_point.get("vector") if isinstance(most_recent_point, dict) else None)
        
        if vector and len(vector) > 0:
            print(f"Found most recent favorite:")
            print(f"   ID: {point_id}")
            print(f"   Image URL: {payload.get('image_url', 'N/A')}")
            print(f"   Timestamp: {payload.get('added_on', 'N/A')}")
            print(f"   Vector size: {len(vector)}D")
            print(f"===== END RECENT FAVORITE CHECK =====\n")
            return vector
        else:
            print(f"No vector found in most recent favorite")
            print(f"===== END RECENT FAVORITE CHECK =====\n")
            return None
        
    except Exception as e:
        print(f"Error retrieving most recent favorite: {e}")
        return None

def search_similar_by_vector(vector: List[float], top_k: int = 10) -> List[Dict]:
    """Search for similar items using a single favorite vector"""
    if client is None:
        return []
    
    print(f"Searching for similar items using recent favorite vector")
    print(f"   Vector size: {len(vector)}D")
    print(f"   First 5 dimensions: {vector[:5]}")
    
    all_results = []
    
    try:
        search_res = client.search(
            collection_name=PRODUCT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )
        
        print(f"Found {len(search_res)} similar items")
        
        for i, r in enumerate(search_res):
            payload = r.payload if hasattr(r, "payload") else (r.get("payload") if isinstance(r, dict) else {})
            item_id = payload.get("id") or payload.get("id_with_ext") or str(getattr(r, "id", ""))
            image_path_in_db = payload.get("image_path") or payload.get("image_url") or ""
            image_url = ""
            
            if image_path_in_db:
                try:
                    path_parts = image_path_in_db.split('/')
                    repo_name = '/'.join(path_parts[:2])
                    image_file = path_parts[-1]
                    image_url = f"https://huggingface.co/datasets/{repo_name}/resolve/main/{image_file}"
                    print(f"   Fixed image URL: {image_path_in_db} → {image_url}")
                except Exception as e:
                    print(f"   Failed to fix image URL {image_path_in_db}: {e}")
                    image_url = image_path_in_db
            else:
                id_with_ext = payload.get("id_with_ext")
                if id_with_ext:
                    image_url = f"https://huggingface.co/datasets/Sharathsoorya/Multimodal_Fashion_Data/resolve/main/{id_with_ext}"
                    print(f"   Constructed image URL from id: {image_url}")
                else:
                    image_url = ""
            
            result_item = {
                "id": item_id,
                "productDisplayName_x": payload.get("productDisplayName_x") or payload.get("product") or "N/A",
                "brandName": payload.get("brandName", "N/A"),
                "subCategory_x": payload.get("subCategory_x") or payload.get("subCategory", "N/A"),
                "baseColour_x": payload.get("baseColour_x") or payload.get("baseColour", "N/A"),
                "price": payload.get("price", "N/A"),
                "rating": payload.get("rating", "N/A"),
                "image_url": image_url,
                "description": payload.get("description", ""),
                "masterCategory": payload.get("masterCategory", "N/A"),
                "season": payload.get("season", "N/A"),
                "usage": payload.get("usage", "N/A"),
                "score": getattr(r, "score", 0.0)
            }
            all_results.append(result_item)
            
            print(f"   {i+1}. {result_item['productDisplayName_x']} (Score: {result_item['score']:.3f})")
                
    except Exception as e:
        print(f"Search failed for vector: {e}")
    
    return all_results

def log_activity(user_email: str, item_vector: List[float], item_id: Optional[str], query_text: str, activity_type: str = "favourite"):
    """Log a purchase/favourite activity in Qdrant."""
    if client is None or not client.collection_exists(FAVOURITES_COLLECTION):
        raise RuntimeError(f"Collection {FAVOURITES_COLLECTION} does not exist or client not initialized")
    
    user_point = get_user_point_by_email(user_email)
    user_id = user_point.payload.get("user_id") if user_point else user_email
    
    point_id = str(uuid.uuid4())
    payload = {
        "user_id": user_id,
        "user_email": user_email,
        "activity_type": activity_type,
        "item_id": item_id or "N/A",
        "query_text": query_text,
        "timestamp": datetime.isoformat()
    }
    
    point = {"id": point_id, "payload": payload}
    if item_vector:
        point["vector"] = item_vector
    
    client.upsert(collection_name=FAVOURITES_COLLECTION, points=[point], wait=True)

def get_recent_favourite_vectors_for_user(user_email: str, days: int = 7) -> List[List[float]]:
    """Return list of vectors for 'favourite' activities by this user."""
    if client is None or not client.collection_exists(FAVOURITES_COLLECTION):
        return []
    
    filt = Filter(must=[
        FieldCondition(key="user_email", match=MatchValue(value=user_email)),
        FieldCondition(key="activity_type", match=MatchValue(value="favourite"))
    ])
    
    points = safe_scroll(FAVOURITES_COLLECTION, filt, limit=200)
    cutoff = datetime.utcnow() - timedelta(days=days)
    vectors = []
    
    for p in points:
        payload = p.payload if hasattr(p, "payload") else p.get("payload", {})
        ts = payload.get("timestamp")
        try:
            if ts and datetime.fromisoformat(ts) < cutoff:
                continue
        except Exception:
            pass
        
        vec = getattr(p, "vector", None) or (p.get("vector") if isinstance(p, dict) else None)
        if vec:
            vectors.append(vec)
    
    return vectors

def get_retrieved_items_local(image_path=None, text=None, top_k=5):
    from retrived import get_retrieved_items
    return get_retrieved_items(image_path=image_path, text=text, top_k=top_k)

def build_rerank_prompt(user_query: str, retrieved_items: List[Dict]) -> str:
    items_summary_lines = []
    for i, item in enumerate(retrieved_items):
        product = item.get("productDisplayName_x") or item.get("product") or "N/A"
        brand = item.get("brandName") or "N/A"
        subcat = item.get("subCategory_x") or "N/A"
        color = item.get("baseColour_x") or "N/A"
        price = item.get("price", "N/A")
        rating = item.get("rating", "N/A")
        desc = (item.get("description") or "")[:140].replace("\n", " ")
        img = item.get("image_path") or item.get("image_url") or ""
        
        items_summary_lines.append(f"Item {i+1}: {product} | Brand: {brand} | Category: {subcat} | Color: {color} | Price: {price} | Rating: {rating} | Image: {img}\n Description: {desc}")
    
    items_summary = "\n".join(items_summary_lines)
    
    prompt = dedent(f"""
    You are an advanced fashion stylist and reranker.
    
    User Query: "{user_query}"
    
    Here are retrieved items (each with metadata):
    {items_summary}
    
    Rerank them by:
    - Query relevance (product/category/color/brand)
    - If budget mentioned, prioritize within budget.
    - Otherwise prefer affordable & high-rated ones.
    
    - Output STRICTLY a Markdown table with:
    | Rank | Product | Gender | MasterCategory | SubCategory | BaseColour | Season | Usage | BrandName | Price | Rating | Image | Why Best |
    
    Include a "Final Recommendation Summary" after the table (4–6 sentences).
    """)
    
    return prompt

def rerank_with_llm(user_query: str, retrieved_results: List[Dict]) -> Tuple[List[Dict], str]:
    if not retrieved_results:
        return [], "<p>No results</p>"
    
    prompt = build_rerank_prompt(user_query, retrieved_results)
    
    try:
        llm_out = call_openrouter(prompt)
        if not llm_out or "|" not in llm_out:
            # Retry with stricter instructions
            llm_out = call_openrouter(prompt + "\n\nReturn strictly as a Markdown table starting with | Rank |")
    except Exception as e:
        print(f"LLM call failed: {e}")
        llm_out = "| Rank | Product | Brand | Category | Color | Price | Rating |\n|------|---------|-------|----------|-------|-------|--------|\n"
        for i, item in enumerate(retrieved_results[:5]):
            product = item.get("productDisplayName_x") or item.get("product") or "N/A"
            brand = item.get("brandName", "N/A")
            category = item.get("subCategory_x", "N/A")
            color = item.get("baseColour_x", "N/A")
            price = item.get("price", "N/A")
            rating = item.get("rating", "N/A")
            llm_out += f"| {i+1} | {product} | {brand} | {category} | {color} | {price} | {rating} |\n"
    
    return retrieved_results, llm_out

def markdown_to_html(md_text: str) -> str:
    import re
    import markdown
    
    # Replace image URLs with viewable links
    img_pattern = re.compile(r'(https?://[^\s\)\|]+?\.(?:jpg|jpeg|png|webp|gif))', re.IGNORECASE)
    md_text = img_pattern.sub(r'[View Image](\1)', md_text)
    
    try:
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        return f"<div class='markdown-table'>{html}</div>"
    except Exception as e:
        print(f"Markdown conversion failed: {e}")
        return f"<div class='markdown-table'><pre>{md_text}</pre></div>"

def user_query_pipeline(user_query: str, image_path: Optional[str] = None, top_k: int = 5, user_email: Optional[str] = None) -> Tuple[List[Dict], str, List[Dict], bool]:
    """
    Enhanced pipeline that handles both primary search and favorite-based recommendations
    
    Returns:
    - primary_results: Results from the main search query
    - llm_html: LLM reranked results in HTML format
    - favorite_suggestions: Similar items based on user's MOST RECENT favorite
    - has_favorites: Boolean indicating if user has favorites
    """
    if client is None:
        print("Qdrant client not initialized - cannot run pipeline")
        return [], "<p>System error: Qdrant not connected</p>", [], False
        
    print(f"\n ===== STARTING PIPELINE =====")
    print(f"Query: {user_query}")
    print(f"User: {user_email}")
    
    primary_results = get_retrieved_items_local(image_path=image_path, text=user_query, top_k=top_k)
    print(f"🔍 Primary results: {len(primary_results)} items")
    
    favorite_suggestions = []
    has_favorites = False

    if user_email:
        try:
            increment_login_count(user_email)
            
            recent_fav_vector = get_most_recent_favourite_vector(user_email)
            
            if recent_fav_vector:
                has_favorites = True
                print(f"User has favorites, getting similar items from MOST RECENT favorite...")
                favorite_suggestions = search_similar_by_vector(recent_fav_vector, top_k=top_k)
                print(f"Found {len(favorite_suggestions)} favorite-based suggestions from recent favorite")
            else:
                print(f"ℹNo favorites found for {user_email}")
                
        except Exception as e:
            print(f"Favorite processing failed: {e}")

    if primary_results:
        _, llm_markdown = rerank_with_llm(user_query, primary_results)
        llm_html = markdown_to_html(llm_markdown)
    else:
        llm_html = "<p>No primary retrieval results.</p>"
    
    print(f"Final results - Primary: {len(primary_results)}, Favorites: {len(favorite_suggestions)}, HasFavorites: {has_favorites}")
    print(f"===== PIPELINE COMPLETE =====\n")
    
    return primary_results, llm_html, favorite_suggestions, has_favorites