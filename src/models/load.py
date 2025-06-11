import torch
from typing import Any, Dict, Optional
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitCLIPTransformerConfig,
    OutfitCLIPTransformer
)
from torch.distributed import get_rank, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP


def load_model(model_type, checkpoint=None, device=None, **cfg_kwargs):
    is_distributed = torch.distributed.is_initialized()

    # Set device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # 분산 학습 환경 설정
    if is_distributed:
        rank = get_rank()
        world_size = get_world_size()
        if torch.backends.mps.is_available():
            map_location = 'mps'
        elif torch.cuda.is_available():
            map_location = f'cuda:{rank}'
        else:
            map_location = 'cpu'
    else:
        rank = 0
        world_size = 1
        if torch.backends.mps.is_available():
            map_location = 'mps'
        elif torch.cuda.is_available():
            map_location = 'cuda'
        else:
            map_location = 'cpu'
    
    # 체크포인트 로드
    state_dict = None
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=map_location)
        cfg = state_dict.get('config', {})
        model_state_dict = state_dict.get('model', {})
    else:
        cfg = cfg_kwargs
        model_state_dict = None
    
    # 모델 초기화
    if model_type == 'original':
        model = OutfitTransformer(OutfitTransformerConfig(**cfg))
    elif model_type == 'clip':
        model = OutfitCLIPTransformer(OutfitCLIPTransformerConfig(**cfg))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.to(device)
    
    # DDP 체크포인트와 일반 체크포인트 호환성 처리
    if model_state_dict:
        new_state_dict = {}
        for k, v in model_state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        if missing:
            print(f"[Warning] Missing keys in state_dict: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys in state_dict: {unexpected}")
        print(f"Loaded model from checkpoint: {checkpoint}")
    
    # DDP 적용 (가중치 로드 후 래핑)
    # MPS doesn't support DDP, so only apply if using CUDA
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], static_graph=True)
    
    return model
