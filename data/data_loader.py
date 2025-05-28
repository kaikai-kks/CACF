import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from .dataset import ShrimpDataset
from config.config import Config

def get_data_loaders():
    all_period_ids = list(range(1, 101))  
    
    np.random.seed(Config.SEED)
    np.random.shuffle(all_period_ids)
    
    train_size = int(len(all_period_ids) * Config.TRAIN_RATIO)
    val_size = int(len(all_period_ids) * Config.VAL_RATIO)
    test_size = len(all_period_ids) - train_size - val_size
    
    train_periods = all_period_ids[:train_size]
    val_periods = all_period_ids[train_size:train_size+val_size]
    test_periods = all_period_ids[train_size+val_size:]
    
    train_dataset = ShrimpDataset(Config.DATA_PATH, train_periods)
    val_dataset = ShrimpDataset(Config.DATA_PATH, val_periods)
    test_dataset = ShrimpDataset(Config.DATA_PATH, test_periods)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
