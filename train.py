import os
import torch
import numpy as np
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import logging
import time
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import Metrics
from data.data_loader import load_dataset, create_dataloader
from config.config import load_config
from models.cacf import CACF
from evaluate import evaluate_model



def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), timestamp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device, clip_grad=None):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_contrastive_loss = 0
    total_alignment_loss = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        visual_data, sensor_data, targets = batch
        
        visual_data = visual_data.to(device)
        sensor_data = sensor_data.to(device)
        targets = targets.to(device)
        
        outputs = model(visual_data, sensor_data, targets)
        
        pred_loss = outputs.get('prediction_loss', 0)
        contrastive_loss = outputs.get('contrastive_loss', 0)
        alignment_loss = outputs.get('alignment_loss', 0)
        
        loss = pred_loss + contrastive_loss + alignment_loss
        
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        if isinstance(pred_loss, torch.Tensor):
            total_pred_loss += pred_loss.item()
        if isinstance(contrastive_loss, torch.Tensor):
            total_contrastive_loss += contrastive_loss.item()
        if isinstance(alignment_loss, torch.Tensor):
            total_alignment_loss += alignment_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_pred_loss = total_pred_loss / len(train_loader)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader)
    avg_alignment_loss = total_alignment_loss / len(train_loader)
    
    return {
        'total': avg_loss,
        'prediction': avg_pred_loss,
        'contrastive': avg_contrastive_loss,
        'alignment': avg_alignment_loss
    }


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_pred_loss = 0
    total_contrastive_loss = 0
    total_alignment_loss = 0
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            visual_data, sensor_data, targets = batch
            
            visual_data = visual_data.to(device)
            sensor_data = sensor_data.to(device)
            targets = targets.to(device)
            
            outputs = model(visual_data, sensor_data, targets)
            
            pred_loss = outputs.get('prediction_loss', 0)
            contrastive_loss = outputs.get('contrastive_loss', 0)
            alignment_loss = outputs.get('alignment_loss', 0)
            
            loss = pred_loss + contrastive_loss + alignment_loss
            
            total_loss += loss.item()
            if isinstance(pred_loss, torch.Tensor):
                total_pred_loss += pred_loss.item()
            if isinstance(contrastive_loss, torch.Tensor):
                total_contrastive_loss += contrastive_loss.item()
            if isinstance(alignment_loss, torch.Tensor):
                total_alignment_loss += alignment_loss.item()
            
            y_pred = outputs['predictions'].detach().cpu()
            y_true = targets.detach().cpu()
            
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    metrics = Metrics.evaluate(all_y_true, all_y_pred, verbose=False)
    
    avg_loss = total_loss / len(val_loader)
    avg_pred_loss = total_pred_loss / len(val_loader)
    avg_contrastive_loss = total_contrastive_loss / len(val_loader)
    avg_alignment_loss = total_alignment_loss / len(val_loader)
    
    loss_dict = {
        'total': avg_loss,
        'prediction': avg_pred_loss,
        'contrastive': avg_contrastive_loss,
        'alignment': avg_alignment_loss
    }
    
    return loss_dict, metrics


def train_model(config, train_loader, val_loader, device, output_dir, logger, writer):
    model = get_model(config)
    model.to(device)
    
    optimizer = get_optimizer(model.parameters(), config)
    
    scheduler_type = config.get("scheduler", {}).get("type", None)
    if scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config["scheduler"].get("factor", 0.1),
            patience=config["scheduler"].get("patience", 10),
            verbose=True
        )
    elif scheduler_type == "cosine_annealing":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["scheduler"].get("T_max", 50),
            eta_min=config["scheduler"].get("eta_min", 0)
        )
    else:
        scheduler = None
    
    num_epochs = config.get("num_epochs", 100)
    early_stopping_patience = config.get("early_stopping_patience", 20)
    clip_grad = config.get("clip_grad", None)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_components": [],
        "val_loss_components": [],
        "val_metrics": [],
        "learning_rates": []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss_dict = train_epoch(model, train_loader, optimizer, None, device, clip_grad)
        
        val_loss_dict, val_metrics = validate(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if scheduler is not None:
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_loss_dict['total'])
            else:
                scheduler.step()
        
        history["train_loss"].append(train_loss_dict['total'])
        history["val_loss"].append(val_loss_dict['total'])
        history["train_loss_components"].append(train_loss_dict)
        history["val_loss_components"].append(val_loss_dict)
        history["val_metrics"].append(val_metrics)
        history["learning_rates"].append(current_lr)
        
        writer.add_scalar('Loss/train_total', train_loss_dict['total'], epoch)
        writer.add_scalar('Loss/val_total', val_loss_dict['total'], epoch)
        
        for component in ['prediction', 'contrastive', 'alignment']:
            if train_loss_dict[component] > 0:
                writer.add_scalar(f'Loss/train_{component}', train_loss_dict[component], epoch)
            if val_loss_dict[component] > 0:
                writer.add_scalar(f'Loss/val_{component}', val_loss_dict[component], epoch)
        
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss_dict['total']:.4f}, "
            f"Val Loss: {val_loss_dict['total']:.4f}, "
            f"Pred Loss: {val_loss_dict['prediction']:.4f}, "
            f"Contrastive Loss: {val_loss_dict['contrastive']:.4f}, "
            f"Alignment Loss: {val_loss_dict['alignment']:.4f}, "
            f"MAE: {val_metrics['MAE']:.4f}, "
            f"RMSE: {val_metrics['RMSE']:.4f}, "
            f"MAPE: {val_metrics['MAPE']:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        if val_loss_dict['total'] < best_val_loss:
            best_val_loss = val_loss_dict['total']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(best_model_state, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered! No improvement in validation loss for {early_stopping_patience} epochs")
                break
        
        if (epoch + 1) % config.get("checkpoint_interval", 10) == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_dict,
                'val_loss': val_loss_dict,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return model, history


def save_training_config(config, output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    
    config_with_timestamp = config.copy()
    config_with_timestamp["timestamp"] = timestamp
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_with_timestamp, f, indent=4)


def save_training_history(history, output_dir):
    history_path = os.path.join(output_dir, "training_history.json")
    
    serializable_history = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "learning_rates": history["learning_rates"],
    }
    
    serializable_history["train_loss_components"] = []
    for comp in history["train_loss_components"]:
        serializable_history["train_loss_components"].append({k: float(v) for k, v in comp.items()})
    
    serializable_history["val_loss_components"] = []
    for comp in history["val_loss_components"]:
        serializable_history["val_loss_components"].append({k: float(v) for k, v in comp.items()})
    
    serializable_history["val_metrics"] = []
    for metrics in history["val_metrics"]:
        serializable_history["val_metrics"].append({k: float(v) for k, v in metrics.items()})
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    return history_path


def main():
    parser = argparse.ArgumentParser(description="CACF model training script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="training_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computing device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    logger, timestamp = setup_logging()
    logger.info(f"Starting training, timestamp: {timestamp}")
    
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    save_training_config(config, output_dir, timestamp)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    logger.info("Loading datasets...")
    train_data = load_dataset(config, split="train")
    val_data = load_dataset(config, split="val")
    test_data = load_dataset(config, split="test")
    
    batch_size = config.get("batch_size", 32)
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    model, history = train_model(config, train_loader, val_loader, device, output_dir, logger, writer)
    
    history_path = save_training_history(history, output_dir)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, device, config, logger)
    
    test_metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=4)
    
    logger.info(f"Test metrics saved to {test_metrics_path}")
    logger.info("Training completed!")
    
    writer.close()


if __name__ == "__main__":
    main()
