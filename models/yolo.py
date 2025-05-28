import torch
import torch.nn as nn
import torchvision
import numpy as np
from config.config import Config
from ultralytics import YOLO

class YOLOFeatureExtractor(nn.Module):
    def __init__(self, model_name=Config.YOLO_MODEL, conf_thres=Config.CONFIDENCE_THRESHOLD, iou_thres=Config.IOU_THRESHOLD):
        super(YOLOFeatureExtractor, self).__init__()
        
        self.model = YOLO(f'yolov8{model_name}.pt')
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    def forward(self, images):
        """
        Args:
            [batch_size, T, M, C, H, W]
        Returns:
            [batch_size, T, 3]
        """
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
                        xyxy = boxes.xyxy.cpu() 
                        
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
