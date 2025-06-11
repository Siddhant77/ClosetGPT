import json
import logging
import os
import pathlib
import pickle
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.logger import get_logger
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz', type=int,
                        default=128)
    parser.add_argument('--n_workers', type=int,
                        default=4)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--device', type=str, 
                        default='mps', choices=['mps', 'cpu'])
    
    return parser.parse_args()


def setup_dataloader(args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    item_dataset = polyvore.PolyvoreItemDataset(
        dataset_dir=args.polyvore_dir, metadata=metadata,
        load_image=False, embedding_dict=embedding_dict
    )
    
    item_dataloader = torch.utils.data.DataLoader(
        dataset=item_dataset, batch_size=args.batch_sz, shuffle=False,
        num_workers=args.n_workers, collate_fn=collate_fn.item_collate_fn
    )

    return item_dataloader


def main(args: Any):  
    # Logging Setup
    logger = get_logger('generate_rec_embeddings', LOGS_DIR)
    logger.info(f'Logger Setup Completed')
    
    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f'Using MPS device')
    else:
        device = torch.device('cpu')
        logger.info(f'MPS not available, using CPU')
    
    # Dataloaders
    item_dataloader = setup_dataloader(args)
    logger.info(f'Dataloader Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()
    logger.info(f'Model Loaded and moved to {device}')
    
    all_ids, all_embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(item_dataloader):
            if args.demo and len(all_embeddings) > 10:
                break
            
            # Move batch data to device if needed
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, list):
                for i, item in enumerate(batch):
                    if hasattr(item, 'to'):
                        batch[i] = item.to(device)
            
            embeddings = model(batch, use_precomputed_embedding=True)  # (batch_size, d_embed)
            
            all_ids.extend([item.item_id for item in batch])
            all_embeddings.append(embeddings.detach().cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Computed {len(all_embeddings)} embeddings")

    # Save numpy array
    save_dir = POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/polyvore_0.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)
    
    logger.info(f"Saved embeddings to {save_path}")
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
