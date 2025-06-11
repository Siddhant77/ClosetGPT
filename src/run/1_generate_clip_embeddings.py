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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, setup
from ..utils.logger import get_logger
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = "{polyvore_dir}/precomputed_clip_embeddings"


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
    parser.add_argument('--world_size', type=int, 
                        default=1)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--device', type=str, 
                        default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    
    return parser.parse_args()


def setup_dataloader(args):
    item_dataset = polyvore.PolyvoreItemDataset(
        dataset_dir=args.polyvore_dir, load_image=True
    )
    
    item_dataloader = torch.utils.data.DataLoader(
        dataset=item_dataset, batch_size=args.batch_sz, shuffle=False,
        num_workers=args.n_workers, collate_fn=collate_fn.item_collate_fn
    )

    return item_dataloader


def compute(args: Any):
    # Set device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Logging Setup
    logger = get_logger('precompute_clip_embedding', LOGS_DIR)
    logger.info(f'Logger Setup Completed')
    logger.info(f'Using device: {device}')
    logger.info(args)

    # Dataloaders
    item_dataloader = setup_dataloader(args)
    logger.info(f'Dataloader Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint, device=device)
    model.eval()
    logger.info(f'Model Loaded')
    
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
            
            embeddings = model.precompute_clip_embedding(batch)
            
            all_ids.extend([item.item_id for item in batch])
            all_embeddings.append(embeddings)
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Computed {len(all_embeddings)} embeddings")

    # Save numpy array
    save_dir = POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/polyvore_0.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)
    
    logger.info(f"Saved embeddings to {save_path}")


if __name__ == '__main__':
    args = parse_args()
    compute(args)
