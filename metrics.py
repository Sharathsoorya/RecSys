import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import json

# Load and preprocess the dataset
def load_dataset():
    # Parse the articleAttributes from string to dict
    def parse_attributes(attr_str):
        try:
            if pd.isna(attr_str) or attr_str in ['-', '{}', '']:
                return {}
            return eval(attr_str) if isinstance(attr_str, str) else attr_str
        except:
            return {}
    
    df = pd.read_csv('Images_csv_metadata.csv')
    df['articleAttributes'] = df['articleAttributes'].apply(parse_attributes)
    return df

# Create ground truth relevance judgments
def create_relevance_judgments(df):
    judgments = {
        "shoes": {
            'criteria': ['articleType=Shoes', 'subCategory=Shoes'],
            'relevant_products': {}
        },
        "dresses": {
            'criteria': ['articleType=Dresses', 'subCategory=Dress'],
            'relevant_products': {}
        },
        "bags": {
            'criteria': ['articleType=Handbags', 'subCategory=Bags', 'masterCategory=Accessories'],
            'relevant_products': {}
        },
        "shirts": {
            'criteria': ['articleType=Shirts', 'subCategory=Topwear'],
            'relevant_products': {}
        },
        "watches": {
            'criteria': ['articleType=Watches', 'subCategory=Watches', 'masterCategory=Accessories'],
            'relevant_products': {}
        },
        "men's casual cotton shirts": {
            'criteria': ['gender=Men', 'articleType=Shirts', 'usage=Casual', 'material=cotton'],
            'relevant_products': {}
        },
        "women's black handbags under 2000": {
            'criteria': ['gender=Women', 'articleType=Handbags', 'baseColour=Black', 'price<2000'],
            'relevant_products': {}
        },
    }
    
    # Score each product for each query
    for query, config in judgments.items():
        for idx, row in df.iterrows():
            score = calculate_relevance_score_criteria(row, config['criteria'], query)
            if score > 0:
                config['relevant_products'][idx] = score
    
    return judgments

def calculate_relevance_score_criteria(product, criteria, query):
    """
    Calculate relevance based on specific criteria for ground truth
    """
    score = 0
    
    for criterion in criteria:
        if 'gender=' in criterion and product.get('gender_x') == criterion.split('=')[1]:
            score += 1
        elif 'articleType=' in criterion and product.get('articleType_x') == criterion.split('=')[1]:
            score += 2  # Higher weight for exact article type match
        elif 'subCategory=' in criterion and product.get('subCategory_x') == criterion.split('=')[1]:
            score += 1
        elif 'masterCategory=' in criterion and product.get('masterCategory_x') == criterion.split('=')[1]:
            score += 1
        elif 'usage=' in criterion and product.get('usage_x') == criterion.split('=')[1]:
            score += 1
        elif 'baseColour=' in criterion and product.get('baseColour_x') == criterion.split('=')[1]:
            score += 1
        elif 'material=cotton' in criterion and 'cotton' in str(product.get('description', '')).lower():
            score += 1
        elif 'price<' in criterion and product.get('price', float('inf')) < float(criterion.split('<')[1]):
            score += 1
    
    return min(score, 3)  # Cap at maximum relevance

def calculate_relevance_score_simple(product, query):
    """
    More flexible relevance scoring that handles schema variations
    For use with actual RAG retrieval results
    """
    score = 0
    query_lower = query.lower()
    
    # Get product fields with fallbacks for schema variations
    product_name = product.get('productDisplayName_x') or product.get('product', '')
    article_type = product.get('articleType_x') or product.get('articleType', '')
    description = product.get('description', '')
    subcategory = product.get('subCategory_x') or product.get('subCategory', '')
    
    # Convert to lowercase for matching
    product_text = f"{product_name} {description} {article_type} {subcategory}".lower()
    
    # Exact category matches (high weight)
    category_terms = {
        'shoes': ['shoe', 'footwear'],
        'dresses': ['dress', 'gown'],
        'bags': ['bag', 'handbag', 'purse'],
        'shirts': ['shirt', 'topwear'],
        'watches': ['watch', 'timepiece']
    }
    
    if query_lower in category_terms:
        terms = category_terms[query_lower]
        for term in terms:
            if term in product_text:
                if term == query_lower or term in [query_lower[:-1], query_lower + 'es']:
                    score += 3  # High relevance for exact match
                else:
                    score += 2  # Partial match
    
    # Field-specific matching
    if query_lower in str(article_type).lower():
        score += 2
    
    if query_lower in str(product_name).lower():
        score += 2
    
    if query_lower in str(subcategory).lower():
        score += 2
    
    if query_lower in str(description).lower():
        score += 1
    
    return min(score, 3)  # Cap at maximum relevance

# Simulate RAG system retrieval (you would replace this with your actual RAG system)
def simulate_rag_retrieval(df, query, top_k=10):
    """
    Simulates a RAG system retrieving products for a query
    In practice, this would be your actual embedding search + LLM reranking
    """
    # Simple keyword-based simulation - replace with your actual RAG
    query_lower = query.lower()
    scores = []
    
    for idx, row in df.iterrows():
        product_text = f"{row.get('productDisplayName_x', '')} {row.get('description', '')} {str(row.get('articleAttributes', {}))}"
        product_text_lower = product_text.lower()
        
        # Simple relevance scoring
        relevance = 0
        for word in query_lower.split():
            if word in product_text_lower:
                relevance += 1
        
        # Boost for exact matches in key fields
        if any(term in str(row.get('articleType_x', '')).lower() for term in query_lower.split()):
            relevance += 2
        if any(term in str(row.get('usage_x', '')).lower() for term in query_lower.split()):
            relevance += 2
        
        scores.append((idx, relevance))
    
    # Return top_k most relevant products
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in scores[:top_k]]

# Calculate nDCG
def calculate_ndcg_for_queries(df, judgments, top_k=10):
    ndcg_scores = {}
    
    for query, config in judgments.items():
        # Get ideal ranking (ground truth)
        ideal_relevance = []
        for product_id, relevance in config['relevant_products'].items():
            ideal_relevance.append(relevance)
        
        # Pad with zeros if needed
        while len(ideal_relevance) < top_k:
            ideal_relevance.append(0)
        ideal_relevance = ideal_relevance[:top_k]
        
        # Get RAG system ranking
        retrieved_products = simulate_rag_retrieval(df, query, top_k)
        
        # Map to relevance scores
        system_relevance = []
        for product_id in retrieved_products:
            system_relevance.append(config['relevant_products'].get(product_id, 0))
        
        # Calculate nDCG
        if len(ideal_relevance) > 0 and max(ideal_relevance) > 0:
            ndcg = ndcg_score([ideal_relevance], [system_relevance], k=top_k)
            ndcg_scores[query] = ndcg
        else:
            ndcg_scores[query] = 0.0
    
    return ndcg_scores

# Debug function to see what's happening
def debug_judgments(judgments, df):
    """Debug the relevance judgments"""
    print("\n🔍 DEBUG: Relevance Judgments")
    print("=" * 60)
    for query, config in judgments.items():
        print(f"\nQuery: '{query}'")
        print(f"Number of relevant products: {len(config['relevant_products'])}")
        if len(config['relevant_products']) > 0:
            print("Top 5 relevant products:")
            top_products = sorted(config['relevant_products'].items(), key=lambda x: x[1], reverse=True)[:5]
            for idx, score in top_products:
                product = df.iloc[idx]
                name = product.get('productDisplayName_x', 'N/A')
                article_type = product.get('articleType_x', 'N/A')
                print(f"  - {name} ({article_type}) - Score: {score}")

# Main evaluation
def evaluate_rag_system():
    df = load_dataset()
    judgments = create_relevance_judgments(df)
    
    # Debug to see what's being considered relevant
    debug_judgments(judgments, df)
    
    ndcg_scores = calculate_ndcg_for_queries(df, judgments, top_k=5)
    
    print("\n" + "=" * 50)
    print("RAG System Evaluation Results (nDCG@5):")
    print("=" * 50)
    for query, score in ndcg_scores.items():
        print(f"{query:.<40} {score:.4f}")
    
    print(f"\nAverage nDCG: {np.mean(list(ndcg_scores.values())):.4f}")
    
    return ndcg_scores, judgments

# Run evaluation
results, judgments = evaluate_rag_system()