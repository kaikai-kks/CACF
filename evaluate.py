import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import logging

from utils.metrics import Metrics
from data.data_loader import load_dataset, create_dataloader
from config.config import load_config


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def evaluate_model(model, test_loader, device, config, logger):
    logger.info("Starting evaluation...")
    
    target_scaler = None
    if hasattr(test_loader.dataset, 'target_scaler'):
        target_scaler = test_loader.dataset.target_scaler
    
    if hasattr(model, 'contrastive_module'):
        logger.info("Detected CACF model, using specialized evaluation...")
        metrics = Metrics.evaluate_cacf_model(model, test_loader, device, target_scaler=target_scaler)
        
        logger.info("Evaluating feature quality...")
        feature_metrics = Metrics.evaluate_feature_quality(model, test_loader, device)
        metrics.update(feature_metrics)
    else:
        metrics = Metrics.evaluate_model(model, test_loader, device, target_scaler=target_scaler)
    
    logger.info("Evaluation results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics


def save_predictions_to_csv(model, test_loader, device, output_dir, logger, target_scaler=None):
    logger.info("Saving predictions to CSV...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            if hasattr(model, 'contrastive_module'):
                visual_data, sensor_data, targets = batch
                
                visual_data = visual_data.to(device)
                sensor_data = sensor_data.to(device)
                targets = targets.to(device)
                
                outputs = model(visual_data, sensor_data)
                y_pred = outputs['predictions'].detach().cpu()
                y_true = targets.detach().cpu()
            elif hasattr(model, 'is_transformer') and model.is_transformer:
                src = batch[0].to(device)
                tgt = batch[1].to(device)
                
                output = model(src, tgt[:, :-1])
                y_pred = output.detach().cpu()
                y_true = tgt[:, 1:].detach().cpu()
            elif hasattr(model, 'is_seq2seq') and model.is_seq2seq:
                src = batch[0].to(device)
                tgt = batch[1].to(device)
                
                y_pred = model.predict(src, tgt.size(1)-1).detach().cpu()
                y_true = tgt[:, 1:].detach().cpu()
            else:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                
                y_pred = model(X).detach().cpu()
                y_true = y.detach().cpu()
            
            if target_scaler is not None:
                y_pred = torch.tensor(target_scaler.inverse_transform(y_pred.numpy()))
                y_true = torch.tensor(target_scaler.inverse_transform(y_true.numpy()))
            
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    timestamps = None
    if hasattr(test_loader.dataset, 'timestamps'):
        timestamps = test_loader.dataset.timestamps
    
    csv_path = os.path.join(output_dir, "predictions.csv")
    Metrics.save_predictions(all_y_true, all_y_pred, timestamps, csv_path)
    
    logger.info(f"Predictions saved to {csv_path}")


def benchmark_performance(model, test_loader, device, output_dir, logger, n_runs=5):
    logger.info(f"Running benchmark with {n_runs} iterations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_results = Metrics.benchmark_model(model, test_loader, device, n_runs)
    
    logger.info(f"Average inference time: {benchmark_results['avg_time']:.6f} seconds")
    logger.info(f"Standard deviation: {benchmark_results['std_time']:.6f} seconds")
    
    benchmark_path = os.path.join(output_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    logger.info(f"Benchmark results saved to {benchmark_path}")
    
    return benchmark_results


def save_feature_embeddings(model, test_loader, device, output_dir, logger):
    logger.info("Extracting feature embeddings...")
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    visual_features = []
    sensor_features = []
    fused_features = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting features"):
            visual_data, sensor_data, batch_targets = batch
            
            visual_data = visual_data.to(device)
            sensor_data = sensor_data.to(device)
            
            outputs = model(visual_data, sensor_data)
            
            batch_visual = outputs['visual_features'][:, -1].detach().cpu().numpy()
            batch_sensor = outputs['sensor_features'][:, -1].detach().cpu().numpy()
            batch_fused = outputs['fused_features'][:, -1].detach().cpu().numpy()
            
            visual_features.append(batch_visual)
            sensor_features.append(batch_sensor)
            fused_features.append(batch_fused)
            targets.append(batch_targets.numpy())
    
    visual_features = np.vstack(visual_features)
    sensor_features = np.vstack(sensor_features)
    fused_features = np.vstack(fused_features)
    targets = np.vstack(targets)
    
    np.save(os.path.join(output_dir, "visual_features.npy"), visual_features)
    np.save(os.path.join(output_dir, "sensor_features.npy"), sensor_features)
    np.save(os.path.join(output_dir, "fused_features.npy"), fused_features)
    np.save(os.path.join(output_dir, "targets.npy"), targets)
    
    logger.info(f"Feature embeddings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--save_features", action="store_true", help="Save feature embeddings for CACF model")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Starting evaluation for model: {args.model_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    logger.info("Loading test dataset...")
    test_data = load_dataset(config, split="test")
    test_loader = create_dataloader(test_data, batch_size=config.get("batch_size", 32), shuffle=False)
    
    target_scaler = None
    if hasattr(test_data, 'target_scaler'):
        target_scaler = test_data.target_scaler
    
    logger.info(f"Loading model from: {args.model_path}")
    model = load_model(config, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    metrics = evaluate_model(model, test_loader, device, config, logger)
    
    save_predictions_to_csv(model, test_loader, device, args.output_dir, logger, target_scaler)
    
    if args.benchmark:
        benchmark_performance(model, test_loader, device, args.output_dir, logger, args.n_runs)
    
    if args.save_features and is_cacf:
        feature_dir = os.path.join(args.output_dir, "feature_embeddings")
        save_feature_embeddings(model, test_loader, device, feature_dir, logger)
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
