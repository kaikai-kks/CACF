import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from ultralytics import YOLO
import numpy as np

class YOLOFeatureExtractor(nn.Module):
    def __init__(self, model_name=Config.YOLO_MODEL, conf_thres=Config.CONFIDENCE_THRESHOLD, iou_thres=Config.IOU_THRESHOLD):
        super(YOLOFeatureExtractor, self).__init__()
        
        self.model = YOLO(f'yolov8{model_name}.pt')
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    def forward(self, images):
        batch_size, T, M, C, H, W = images.shape
        
        day_features = torch.zeros((batch_size, T, 3), device=self.device)
        
        for b in range(batch_size):
            for t in range(T):
                day_length = []
                day_width = []
                day_density = []
                
                for m in range(M):
                    img = images[b, t, m].cpu().numpy()  
                    img = img.transpose(1, 2, 0)  
                    
                    results = self.model(img, conf=self.conf_thres, iou=self.iou_thres, device=self.device)
                    
                    boxes = results[0].boxes
                    
                    if len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu()  # [x1, y1, x2, y2]
                        
                        margin = 10  
                        valid_detections = []
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = xyxy[i]
                            if x1 > margin and y1 > margin and x2 < W - margin and y2 < H - margin:
                                valid_detections.append(xyxy[i])
                        
                        if len(valid_detections) > 0:
                            valid_detections = torch.stack(valid_detections)
                            
                            lengths = torch.sqrt((valid_detections[:, 2] - valid_detections[:, 0])**2 + 
                                                (valid_detections[:, 3] - valid_detections[:, 1])**2)
                            widths = torch.min(valid_detections[:, 2] - valid_detections[:, 0],
                                              valid_detections[:, 3] - valid_detections[:, 1])
                            
                            avg_length = lengths.mean().item()
                            avg_width = widths.mean().item()
                            
                            density = len(valid_detections) / (H * W)
                            
                            day_length.append(avg_length)
                            day_width.append(avg_width)
                            day_density.append(density)
                
                if len(day_length) > 0:
                    day_features[b, t, 0] = np.mean(day_length)
                    day_features[b, t, 1] = np.mean(day_width)
                    day_features[b, t, 2] = np.mean(day_density)
        
        return day_features


class VisualGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=Config.GRU_HIDDEN_SIZE, num_layers=Config.GRU_NUM_LAYERS, 
                 dropout=Config.DROPOUT_RATE, feature_extractor=None):
        super(VisualGRU, self).__init__()
        
        self.feature_extractor = feature_extractor if feature_extractor else YOLOFeatureExtractor()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, images, initial_hidden=None):
        """
        Args:
            [batch_size, T, M, C, H, W]
        Returns:
            [batch_size, T, 1]
        """
        batch_size = images.shape[0]
        
        features = self.feature_extractor(images)  # [batch_size, T, 3]
        
        if initial_hidden is None:
            outputs, hidden = self.gru(features)
        else:
            outputs, hidden = self.gru(features, initial_hidden)
        
        outputs = self.dropout(outputs)
        
        outputs = self.fc(outputs)  # [batch_size, T, 1]
        
        return outputs, hidden
    
    def predict(self, images):
        """
        Args:
            [batch_size, T, M, C, H, W]
        Returns:
            [batch_size, T]
        """
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(images)
            predictions = outputs.squeeze(-1)
        return predictions


class VisualGRUWithAttention(VisualGRU):
    def __init__(self, input_size=3, hidden_size=Config.GRU_HIDDEN_SIZE, num_layers=Config.GRU_NUM_LAYERS, 
                 dropout=Config.DROPOUT_RATE, feature_extractor=None):
        super(VisualGRUWithAttention, self).__init__(
            input_size, hidden_size, num_layers, dropout, feature_extractor
        )
        
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, images, initial_hidden=None):
        """
        Args:
            [batch_size, T, M, C, H, W]
        Returns:
            [batch_size, T, 1]
        """
        batch_size = images.shape[0]
        
        features = self.feature_extractor(images)  # [batch_size, T, 3]
        
        if initial_hidden is None:
            gru_outputs, hidden = self.gru(features)
        else:
            gru_outputs, hidden = self.gru(features, initial_hidden)
        
        attention_weights = F.softmax(self.attention(gru_outputs), dim=1)  # [batch_size, T, 1]
        context = torch.sum(attention_weights * gru_outputs, dim=1)  # [batch_size, hidden_size]
        
        context = self.dropout(context)
        
        output = self.fc(context).unsqueeze(1)  # [batch_size, 1, 1]
        
        outputs = output.expand(-1, features.size(1), -1)  # [batch_size, T, 1]
        
        return outputs, hidden
