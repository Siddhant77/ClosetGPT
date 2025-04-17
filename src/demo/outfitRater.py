import os
import sys
import gc  # For garbage collection

# Add the base src directory to the path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(SRC_DIR)

import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from src.data.datatypes import FashionItem, FashionCompatibilityQuery
from itertools import product
from src.data.datasets.polyvore import load_metadata, load_item

# Add the absolute import for loading the model
from src.models.load import load_model


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
    
    # Only process 'tops' and 'bottoms' categories
    for category in ['tops', 'bottoms','shoes']:
        if category in categorized_items:
            items_in_category = []
            # Limit the number of items loaded per category
            limited_item_ids = categorized_items[category][:max_items_per_category]
            
            for item_id in limited_item_ids:
                item = load_item(dataset_dir, metadata, item_id, load_image, embedding_dict)
                item.metadata = metadata.get(item_id, {})  # Add metadata to each item
                items_in_category.append(item)
                
            categorized_data[category] = items_in_category
            
            # Force garbage collection after each category
            gc.collect()
    
    return categorized_data

# Function to organize and load the data by category
def organize_and_load_data(dataset_dir: str, load_image: bool = False, embedding_dict: dict = None, 
                          max_items_per_category: int = 30) -> Dict[str, List[FashionItem]]:
    # Organize items by categories
    categorized_items, metadata = organize_metadata_by_category(dataset_dir)
    
    # Load items for tops and bottoms categories only
    categorized_data = load_items_by_category(
        dataset_dir, categorized_items, metadata, 
        load_image, embedding_dict, max_items_per_category
    )
    
    return categorized_data


# Function to compute compatibility score for a list of items
@torch.no_grad()
def compute_compatibility_score(outfit: List[FashionItem], model) -> float:
    # Create a compatibility query instead of using PolyvoreCompatibilityDataset directly
    query = FashionCompatibilityQuery(outfit=outfit)
    score = model.predict_score(query=[query], use_precomputed_embedding=False)[0].detach().cpu()
    return float(score)

# Function to generate combinations with only tops and bottoms
def generate_outfit_combinations_and_scores(categorized_data: Dict[str, List[FashionItem]], model, max_combinations: int = 200):
    # Verify both required categories exist
    if 'tops' not in categorized_data or 'bottoms' not in categorized_data:
        print("Error: Both 'tops' and 'bottoms' categories are required")
        return []
    
    # Get top and bottom items
    tops = categorized_data['tops']
    bottoms = categorized_data['bottoms']
    
    # Generate combinations of tops and bottoms only
    all_combinations = list(product(tops, bottoms))
    
    # Limit the total number of combinations
    limited_combinations = all_combinations[:max_combinations]
    print(f"Generated {len(limited_combinations)} top+bottom outfit combinations.")
    
    scores = []
    
    # For each combination of items, compute the compatibility score
    for combination in limited_combinations:
        score = compute_compatibility_score(list(combination), model)
        scores.append({
            'outfit': combination,
            'score': score
        })
        
        # Force garbage collection periodically
        if len(scores) % 10 == 0:
            gc.collect()
    
    return scores


if __name__ == '__main__':
    dataset_dir = '/Users/atharva/Documents/TAMU/Spring 2025/Information Search Retreival CSCE670/ClosetGpt/outfit-transformer/datasets/polyvore'  #replace with path to polyvore dataset
    
    # Load the model
    model = load_model(model_type='clip', checkpoint='/Users/atharva/Documents/TAMU/Spring 2025/Information Search Retreival CSCE670/ClosetGpt/outfit-transformer/checkpoints/compatibillity_clip_best.pth') #replace with path to model checkpoint
    model.eval()

    # Process with reasonable limits
    max_items_per_category = 300  # Load up to 30 items per category
    max_combinations = 2000       # Limit combinations to test (900 possible with 30x30)
    
    # Organize and load the dataset with limits (tops and bottoms only)
    categorized_data = organize_and_load_data(
        dataset_dir, 
        load_image=True,
        max_items_per_category=max_items_per_category
    )
    
    # Generate top+bottom combinations and compute scores
    outfit_scores = generate_outfit_combinations_and_scores(
        categorized_data, 
        model,
        max_combinations=max_combinations
    )
    
    # Sort by score from highest to lowest
    outfit_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Print out the scores for the outfits with detailed item information
    for i, outfit_data in enumerate(outfit_scores):
        print(f"\n===== Outfit {i+1} (Score: {outfit_data['score']:.4f}) =====")
        
        # Print details for each item in the outfit
        for j, item in enumerate(outfit_data['outfit']):
            category = item.category.upper() if item.category else "UNKNOWN"
            description = item.description if item.description else "No description"
            print(f"  {j+1}. [{category}] {description}")
        
        # Add a visual separator between outfits
        print("-" * 50)
        
        # Only show top 20 outfits
        if i >= 19:
            break