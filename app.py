# main.py (your FastAPI app)
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import HfApi
from qdrant_client import QdrantClient, models
from uuid import uuid4
from embed_favourite import find_item_by_url
from data_valid import get_user_by_email
from rerank_llm import user_query_pipeline, set_qdrant_client  

app = FastAPI(title="FashionRAG - Multimodal Fashion Search")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

QDRANT_URL = "http://localhost:6333"
REGISTER_COLLECTION = "register_collection"
FAVOURITES_COLLECTION = "favourites_collection"

qdrant = QdrantClient(url=QDRANT_URL)

set_qdrant_client(qdrant)
print("Shared Qdrant client setup complete")

if REGISTER_COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=REGISTER_COLLECTION,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    print(f"Created Qdrant collection: {REGISTER_COLLECTION}")

if FAVOURITES_COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=FAVOURITES_COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )
    print(f"Created Qdrant collection: {FAVOURITES_COLLECTION}")

REPO_ID = "Sharathsoorya/Multimodal_Fashion_Data"
api = HfApi()
print("Fetching file list from Hugging Face...")
_ = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

metadata_df = pd.read_csv("Images_csv_metadata.csv")
metadata_df["id_with_ext"] = metadata_df["id_with_ext"].astype(str)
metadata_df["image_url"] = metadata_df["id_with_ext"].apply(
    lambda x: f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{x}"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    email = request.cookies.get("user")
    items = metadata_df.head(20).to_dict(orient="records")

    has_favorites = False
    if email:
        from rerank_llm import get_most_recent_favourite_vector
        recent_fav_vector = get_most_recent_favourite_vector(email)
        has_favorites = recent_fav_vector is not None
        print(f"🏠 Home page - User: {email}, Has favorites: {has_favorites}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": email,
        "items": items,
        "query": "",
        "has_favorites": has_favorites
    })

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "register_error": "Passwords do not match."
        })
    
    try:
        # store in Qdrant
        qdrant.upsert(
            collection_name=REGISTER_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector=[0.0, 0.0, 0.0],  
                    payload={
                        "email": email,
                        "name": name,
                        "password": password,
                        "created_at": datetime.now().isoformat()
                    }
                )
            ]
        )
        print(f"Registered user in Qdrant: {email}")
    except Exception as e:
        print(f"Failed to store user in Qdrant: {e}")
    
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("user", email)
    return response

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "login_error": "User not found."
        })
    
    payload = user.payload if hasattr(user, "payload") else user.get("payload", {})
    if payload.get("password") != password:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "login_error": "Invalid email or password."
        })
    
    print(f"Logged in: {email}")
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("user", email)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("user")
    return response

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(""),
    file: UploadFile = File(None)
):
    email = request.cookies.get("user")
    if not email:
        return RedirectResponse("/", status_code=303)
    
    print(f"Search initiated by {email}, Query: {query}")
 
    image_path = None
    if file and file.filename:
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Uploaded image saved to: {image_path}")

    primary_results, comparison_table, fav_suggestions, has_favorites = user_query_pipeline(
        user_query=query,
        image_path=image_path,
        top_k=10,
        user_email=email
    )

    if image_path and os.path.exists(image_path):
        os.remove(image_path)
        if os.path.exists(os.path.dirname(image_path)):
            os.rmdir(os.path.dirname(image_path))
        print(f"🧹 Cleaned up temporary file: {image_path}")
    
    print(f"Search results - Primary: {len(primary_results)}, Favorites: {len(fav_suggestions)}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": email,
        "items": primary_results,
        "query": query,
        "comparison_table": comparison_table,
        "fav_suggestions": fav_suggestions,
        "has_favorites": has_favorites
    })

@app.post("/favourite", response_class=HTMLResponse)
async def add_favourite(request: Request, image_url: str = Form(...)):
    email = request.cookies.get("user")
    if not email:
        return RedirectResponse("/", status_code=303)
    
    print(f"Favourite clicked by {email}")
    print(f"Image URL received: {image_url}")
 
    item = find_item_by_url(image_url=image_url)
    if not item:
        print(f"No matching item found for URL: {image_url}")
        return RedirectResponse("/", status_code=303)
    
    vector = item.get("vector")
    if not vector or len(vector) != 1024:
        print("Invalid or missing vector in item")
        return RedirectResponse("/", status_code=303)
    
    point_id = str(uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    qdrant.upsert(
        collection_name=FAVOURITES_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "email": email,
                    "image_url": image_url,
                    "added_on": timestamp
                }
            )
        ]
    )
    
    print(f"Stored favourite in Qdrant [{FAVOURITES_COLLECTION}] with ID: {point_id}")
    print(f"Payload: {{'email': '{email}', 'image_url': '{image_url}', 'added_on': '{timestamp}'}}")
    print(f"Vector size: {len(vector)}D")
    
    try:
        verify_points = qdrant.retrieve(
            collection_name=FAVOURITES_COLLECTION,
            ids=[point_id]
        )
        if verify_points:
            print(f"Verification: Successfully stored and retrieved favorite")
        else:
            print(f"Verification: Failed to retrieve stored favorite")
    except Exception as e:
        print(f"Verification failed: {e}")
    
    return RedirectResponse("/", status_code=303)

@app.post("/buy", response_class=HTMLResponse)
async def buy(request: Request, product: str = Form(...)):
    email = request.cookies.get("user")
    if not email:
        return RedirectResponse("/", status_code=303)
    
    print(f"🛒 Buy clicked by {email} for product: {product}")
    return RedirectResponse("/", status_code=303)

@app.get("/debug/recent-favorite")
async def debug_recent_favorite(request: Request):
    email = request.cookies.get("user")
    if not email:
        return {"error": "Not logged in"}
    
    from rerank_llm import get_most_recent_favourite_vector
    
    recent_vector = get_most_recent_favourite_vector(email)
    
    return {
        "email": email,
        "has_recent_favorite": recent_vector is not None,
        "vector_size": len(recent_vector) if recent_vector else 0,
        "status": "success" if recent_vector else "no_favorites_found"
    }

@app.get("/debug/all-favorites")
async def debug_all_favorites(request: Request):
    email = request.cookies.get("user")
    if not email:
        return {"error": "Not logged in"}
    
    try:
        filt = models.Filter(
            must=[models.FieldCondition(key="email", match=models.MatchValue(value=email))]
        )
        points, _ = qdrant.scroll(
            collection_name=FAVOURITES_COLLECTION,
            scroll_filter=filt,
            limit=50,
            with_payload=True,
            with_vectors=False
        )
        
        favorites_info = []
        for p in points:
            payload = p.payload if hasattr(p, "payload") else p.get("payload", {})
            favorites_info.append({
                "id": getattr(p, "id", "unknown"),
                "image_url": payload.get("image_url", "N/A"),
                "added_on": payload.get("added_on", "N/A"),
                "timestamp_parsed": "yes" if payload.get("added_on") else "no"
            })
        
        return {
            "email": email,
            "total_favorites": len(favorites_info),
            "favorites": favorites_info
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/collections")
async def debug_collections():
    try:
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        return {
            "collections": collection_names,
            "total_collections": len(collection_names)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/favorites-count/{email}")
async def debug_favorites_count(email: str):
    try:
        filt = models.Filter(
            must=[models.FieldCondition(key="email", match=models.MatchValue(value=email))]
        )
        count_result = qdrant.count(
            collection_name=FAVOURITES_COLLECTION,
            count_filter=filt
        )
        return {
            "email": email,
            "favorites_count": count_result.count,
            "collection": FAVOURITES_COLLECTION
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/test-similarity/{email}")
async def debug_test_similarity(email: str):
    try:
        from rerank_llm import get_most_recent_favourite_vector, search_similar_by_vector
        
        recent_vector = get_most_recent_favourite_vector(email)
        if not recent_vector:
            return {"error": "No recent favorite found"}
        
        similar_items = search_similar_by_vector(recent_vector, top_k=5)
        
        return {
            "email": email,
            "recent_vector_size": len(recent_vector),
            "similar_items_found": len(similar_items),
            "similar_items": [
                {
                    "product": item.get("productDisplayName_x", "N/A"),
                    "brand": item.get("brandName", "N/A"),
                    "score": round(item.get("score", 0), 3)
                }
                for item in similar_items
            ]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)