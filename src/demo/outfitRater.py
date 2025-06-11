import os
import sys
import gc  # For garbage collection

# Add the base src directory to the path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(SRC_DIR)

polyvore_dir = SRC_DIR + '/datasets/polyvore'
model_path = SRC_DIR +  '/checkpoints/compatibility_clip_best.pth'
OUTFITS_PATH = SRC_DIR + '/datasets/outfits_v1.json'


from tqdm import tqdm

import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from src.data.datatypes import FashionItem, FashionCompatibilityQuery, Outfit
from itertools import product
from src.data.datasets.polyvore import load_metadata, load_item

from sklearn.cluster import KMeans
import json
import os
import pandas as pd
import random
import numpy as np

# Add the absolute import for loading the model
from src.models.load import load_model



def cluster_outfits_and_select_top(outfits: List[Outfit], n_clusters : int = 5):
    print(f"Clustering {len(outfits)} outfits into {n_clusters} clusters.")
    # Safely extract and average each outfit's embeddings
    embeddings = []
    for outfit in outfits:
        emb = outfit.embedding # entry['embeddings']
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        if emb.ndim == 3:
            emb = emb.mean(axis=1).squeeze()  # average over item dimension if needed
        if emb.ndim == 2:
            emb = emb.mean(axis=0)  # final fallback, average over item axis
        embeddings.append(emb)
    
    embeddings = np.stack(embeddings)  # Shape: (num_outfits, emb_dim)
    scores = np.array([outfit.score for outfit in outfits])

    print("Embeddings shape after pooling:", embeddings.shape)

    # Cluster with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    print("Finished KMeans clustering.")

    # Group outfits by cluster and select top scoring one from each
    cluster_tops = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            print(f"Warning: Cluster {cluster_id} has no outfits.")
            continue
        cluster_entries = [outfits[i] for i in cluster_indices]
    
        # top_entry = max(cluster_entries, key=lambda x: x.score)
    
        top_5 = sorted(cluster_entries, key=lambda x: x.score, reverse=True)[:5]
        chosen = random.choice(top_5)
        cluster_tops.append(chosen)
        print(f"Selected outfit from cluster {cluster_id} with score {chosen.score:.4f}")

    return cluster_tops

# Function to organize metadata by category
def organize_metadata_by_category(dataset_dir: str) -> Dict[str, List[str]]:
    metadata = load_metadata(dataset_dir)
    
    # Organizing items by their category
    categorized_items = defaultdict(list)
    for item_id, item_data in metadata.items():
        category = item_data.get('semantic_category', 'Uncategorized')
        categorized_items[category].append(item_id)
    
    return categorized_items, metadata

# Function to load items based on categories
def load_items_by_category(dataset_dir: str, categorized_items: Dict[str, List[str]], metadata: dict, 
                          load_image: bool = False, embedding_dict: dict = None, 
                          max_items_per_category: int = 30) -> Dict[str, List[FashionItem]]:
    categorized_data = {}
    print(f"Categorized items: {len(categorized_items)}")
    # Only process 'tops' and 'bottoms' categories
    for category in ['tops', 'bottoms','shoes']:
        if category in categorized_items:
            items_in_category = []
            # Limit the number of items loaded per category
            limited_item_ids = categorized_items[category][:max_items_per_category]

            print(f"Loading {len(limited_item_ids)} items for category '{category}'")
            
            for item_id in limited_item_ids:
                item = load_item(dataset_dir, metadata, item_id, load_image, embedding_dict)
                item.metadata = metadata.get(item_id, {})  # Add metadata to each item
                items_in_category.append(item)
                
            categorized_data[category] = items_in_category
            
            # Force garbage collection after each category
            gc.collect()
    
    return categorized_data

# Function to organize and load the data by category
def organize_and_load_data(
        dataset_dir: str, 
        load_image: bool = False, 
        embedding_dict: dict = None, 
        max_items_per_category: int = 30
        ) -> Dict[str, List[FashionItem]]:
    # Organize items by categories
    print("Organizing metadata by category...")
    categorized_items, metadata = organize_metadata_by_category(dataset_dir)
    print("Items organized by metadata.")
    
    # Load items for tops and bottoms categories only
    print("Loading items by category...")
    categorized_data = load_items_by_category(
        dataset_dir, categorized_items, metadata, 
        load_image, embedding_dict, max_items_per_category
    )
    print("Items loaded by category.")
    
    return categorized_data


# Function to compute compatibility score for a list of items
@torch.no_grad()
def compute_compatibility_score(
    outfit: List[FashionItem], 
    model
    ) -> float:
    
    # Create a compatibility query instead of using PolyvoreCompatibilityDataset directly
    query = FashionCompatibilityQuery(outfit=outfit)
    scores, embeddings = model.predict_score_embeddings(query=[query], use_precomputed_embedding=False)
    score = scores[0].item()
    embedding = embeddings[0].detach().cpu().numpy()

    return float(score), embedding

# Function to generate combinations with only tops and bottoms
def generate_outfit_combinations_and_scores(
        categorized_data: Dict[str, List[FashionItem]], 
        model, 
        max_combinations: int = 200
        ) -> List[Outfit]:
    # Verify both required categories exist
    if 'tops' not in categorized_data or 'bottoms' not in categorized_data:
        print("Error: Both 'tops' and 'bottoms' categories are required")
        return []
    
    # Get top and bottom items
    tops = categorized_data['tops']
    bottoms = categorized_data['bottoms']
    shoes = categorized_data.get('shoes', [])
    
    # Generate combinations of tops and bottoms only
    all_combinations = list(product(tops, bottoms))
    
    # Limit the total number of combinations
    limited_combinations = all_combinations[:max_combinations]
    print(f"Generated {len(limited_combinations)} top+bottom outfit combinations.")
    
    outfits = []
    # For each combination of items, compute the compatibility score
    for combination in tqdm(limited_combinations, desc="Scoring outfits"):
        score, embedding = compute_compatibility_score(list(combination), model)
        outfits.append(Outfit(fashion_items=combination, score=score, embedding=embedding))        
        # Force garbage collection periodically
        if len(outfits) % 10 == 0:
            gc.collect()
    
    print(f"Scored {len(outfits)} outfits.")
    return outfits

def get_top_outfits(outfits: list[Outfit]) -> list[Outfit]:
    # Sort by score from highest to lowest
    print("Sorting outfits by score...")
    outfits.sort(key=lambda x: x.score, reverse=True)

    print("Clustering outfits to select top ones...")
    clusterTops = cluster_outfits_and_select_top(outfits=outfits, n_clusters=8)
    print(f"Selected {len(clusterTops)} top outfits from clusters.")

    return clusterTops

def generate_outfits_and_scores():
    
    print("--- Starting Outfit Generation and Scoring ---")
    # Load the model
    print("Loading model...")
    model = load_model(model_type='clip', checkpoint=model_path)
    model.eval()
    print("Model loaded.")

    # Process with reasonable limits
    max_items_per_category = 5
    max_combinations = 200
    print(f"Max items per category: {max_items_per_category}, Max combinations: {max_combinations}")
    
    # Organize and load the dataset with limits (tops and bottoms only)
    print("Organizing and loading data...")
    categorized_data = organize_and_load_data(
        polyvore_dir, 
        load_image=True,
        max_items_per_category=max_items_per_category
    )
    print("Data organized and loaded.")

    # Generate top+bottom combinations and compute scores
    print("Generating and scoring outfit combinations...")
    outfits = generate_outfit_combinations_and_scores(
        categorized_data, 
        model,
        max_combinations=max_combinations
    )
    print("Finished generating and scoring outfits.")

    # Print out the scores for the outfits with detailed item information
    print("\n--- Top 20 Generated Outfits ---")
    for i, outfit in enumerate(outfits):
        if i >= 20:
            break
        print(f"\n===== Outfit {i+1} (Score: {outfit.score:.4f}) =====")
        
        # Print details for each item in the outfit
        for j, item in enumerate(outfit.fashion_items):
            category = item.category.upper() if item.category else "UNKNOWN"
            description = item.description if item.description else "No description"
            print(f"  {j+1}. [{category}] {description} ")
        
        # Add a visual separator between outfits
        print("-" * 50)
    
    # No longer saving to file, returning directly
    print("--- Outfit Generation and Scoring Complete ---")
    return outfits

generate_outfits_and_scores()