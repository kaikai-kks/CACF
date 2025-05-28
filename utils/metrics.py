import numpy as np
import torch
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time


class Metrics:
    
    def mae(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        return mean_absolute_error(y_true, y_pred)
    
    def rmse(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mape(y_true, y_pred, epsilon=1e-8):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        mask = np.abs(y_true) > epsilon
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + epsilon))) * 100
    
    def evaluate(y_true, y_pred, verbose=True):
        metrics = {
            'MAE': Metrics.mae(y_true, y_pred),
            'RMSE': Metrics.rmse(y_true, y_pred),
            'MAPE': Metrics.mape(y_true, y_pred)
        }
        
        if verbose:
            print("Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_cacf_model(model, dataloader, device, target_scaler=None):

        model.eval()
        all_y_true = []
        all_y_pred = []
        all_contrastive_losses = []
        all_alignment_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                visual_data, sensor_data, targets = batch
                
                visual_data = visual_data.to(device)
                sensor_data = sensor_data.to(device)
                targets = targets.to(device)
                
                outputs = model(visual_data, sensor_data)
                
                y_pred = outputs['predictions'].detach().cpu()
                y_true = targets.detach().cpu()
                
                if 'contrastive_loss' in outputs:
                    all_contrastive_losses.append(outputs['contrastive_loss'].item())
                
                if 'alignment_error' in outputs:
                    all_alignment_errors.append(outputs['alignment_error'].item())
                
                if target_scaler is not None:
                    y_pred = torch.tensor(target_scaler.inverse_transform(y_pred.numpy()))
                    y_true = torch.tensor(target_scaler.inverse_transform(y_true.numpy()))
                
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
        
        all_y_true = torch.cat(all_y_true, dim=0).numpy()
        all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
        
        metrics = Metrics.evaluate(all_y_true, all_y_pred, verbose=True)
        
        if all_contrastive_losses:
            avg_contrastive_loss = np.mean(all_contrastive_losses)
            metrics['Contrastive_Loss'] = avg_contrastive_loss
            print(f"Average Contrastive Loss: {avg_contrastive_loss:.4f}")
        
        if all_alignment_errors:
            avg_alignment_error = np.mean(all_alignment_errors)
            metrics['Alignment_Error'] = avg_alignment_error
            print(f"Average Alignment Error: {avg_alignment_error:.4f}")
        
        return metrics

    def evaluate_model(model, dataloader, device, scaler=None, target_scaler=None):
        if hasattr(model, 'contrastive_module'):
            return Metrics.evaluate_cacf_model(model, dataloader, device, target_scaler)
        
        model.eval()
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for batch in dataloader:
                if hasattr(model, 'is_transformer') and model.is_transformer:
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
        
        return Metrics.evaluate(all_y_true, all_y_pred, verbose=True)
    
    def benchmark_model(model, dataloader, device, n_runs=5):
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for _ in range(n_runs):
                batch_times = []
                
                for batch in dataloader:
                    if hasattr(model, 'contrastive_module'):
                        visual_data, sensor_data, _ = batch
                        visual_data = visual_data.to(device)
                        sensor_data = sensor_data.to(device)
                        
                        start_time = time.time()
                        _ = model(visual_data, sensor_data)
                        end_time = time.time()
                    elif hasattr(model, 'is_transformer') and model.is_transformer:
                        src = batch[0].to(device)
                        tgt = batch[1][:, :-1].to(device)
                        
                        start_time = time.time()
                        _ = model(src, tgt)
                        end_time = time.time()
                    elif hasattr(model, 'is_seq2seq') and model.is_seq2seq:
                        src = batch[0].to(device)
                        
                        start_time = time.time()
                        _ = model.predict(src, tgt.size(1)-1)  
                        end_time = time.time()
                    else:
                        X = batch[0].to(device)
                        
                        start_time = time.time()
                        _ = model(X)
                        end_time = time.time()
                    
                    batch_times.append(end_time - start_time)
                
                inference_times.append(np.mean(batch_times))
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        print(f"Benchmark Results (over {n_runs} runs):")
        print(f"Average Inference Time per Batch: {avg_time:.6f} seconds")
        print(f"Standard Deviation: {std_time:.6f} seconds")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'runs': n_runs
        }
    
    def save_predictions(y_true, y_pred, timestamps=None, save_path='predictions.csv'):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        if timestamps is not None:
            df = pd.DataFrame({
                'Timestamp': timestamps,
                'True': y_true.flatten(),
                'Predicted': y_pred.flatten()
            })
        else:
            df = pd.DataFrame({
                'True': y_true.flatten(),
                'Predicted': y_pred.flatten()
            })
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    
    def compare_models(models, names, dataloader, device, scaler=None, target_scaler=None):
        results = []
        
        for model, name in zip(models, names):
            print(f"Evaluating {name}...")
            metrics = Metrics.evaluate_model(model, dataloader, device, scaler, target_scaler)
            metrics['Model'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        cols = df.columns.tolist()
        cols.remove('Model')
        cols = ['Model'] + cols
        df = df[cols]
        
        return df
    
    def evaluate_feature_quality(model, dataloader, device):

        if not hasattr(model, 'contrastive_module'):
            print("Model does not have contrastive module, cannot evaluate feature quality.")
            return {}
        
        model.eval()
        
        intra_modal_similarities = {
            'visual': [],
            'sensor': []
        }
        
        cross_modal_similarities = []
        
        with torch.no_grad():
            for batch in dataloader:
                visual_data, sensor_data, _ = batch
                
                visual_data = visual_data.to(device)
                sensor_data = sensor_data.to(device)
                
                outputs = model(visual_data, sensor_data)
                
                visual_features = outputs.get('visual_features', None)
                sensor_features = outputs.get('sensor_features', None)
                
                if visual_features is not None and sensor_features is not None:
                    batch_size = visual_features.size(0)
                    
                    for i in range(batch_size):
                        for j in range(i+1, batch_size):
                            sim = torch.nn.functional.cosine_similarity(
                                visual_features[i].view(1, -1), 
                                visual_features[j].view(1, -1)
                            ).item()
                            intra_modal_similarities['visual'].append(sim)
                    
                    for i in range(batch_size):
                        for j in range(i+1, batch_size):
                            sim = torch.nn.functional.cosine_similarity(
                                sensor_features[i].view(1, -1), 
                                sensor_features[j].view(1, -1)
                            ).item()
                            intra_modal_similarities['sensor'].append(sim)
                    
                    for i in range(batch_size):
                        sim = torch.nn.functional.cosine_similarity(
                            visual_features[i].view(1, -1), 
                            sensor_features[i].view(1, -1)
                        ).item()
                        cross_modal_similarities.append(sim)
        
        metrics = {}
        
        if intra_modal_similarities['visual']:
            metrics['Visual_Intra_Similarity'] = np.mean(intra_modal_similarities['visual'])
            print(f"Average Visual Intra-modal Similarity: {metrics['Visual_Intra_Similarity']:.4f}")
        
        if intra_modal_similarities['sensor']:
            metrics['Sensor_Intra_Similarity'] = np.mean(intra_modal_similarities['sensor'])
            print(f"Average Sensor Intra-modal Similarity: {metrics['Sensor_Intra_Similarity']:.4f}")
        
        if cross_modal_similarities:
            metrics['Cross_Modal_Similarity'] = np.mean(cross_modal_similarities)
            print(f"Average Cross-modal Similarity: {metrics['Cross_Modal_Similarity']:.4f}")
        
        if 'Cross_Modal_Similarity' in metrics and 'Visual_Intra_Similarity' in metrics and 'Sensor_Intra_Similarity' in metrics:
            intra_avg = (metrics['Visual_Intra_Similarity'] + metrics['Sensor_Intra_Similarity']) / 2
            alignment_quality = metrics['Cross_Modal_Similarity'] / intra_avg
            metrics['Alignment_Quality'] = alignment_quality
            print(f"Alignment Quality: {alignment_quality:.4f}")
        
        return metrics
