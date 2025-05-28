import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class ShrimpDataset(Dataset):
    def __init__(self, data_path, period_list, transform=None):
        """        
        Args:
            data_path
            period_list
            transform
        """
        self.data_path = data_path
        self.period_list = period_list
        self.transform = transform
        
        self.visual_data = {}  
        self.sensor_data = {}  
        self.biomass = {}     
        
        self._load_data()
    
    def _load_data(self):
        for period_id in self.period_list:
            period_visual_path = os.path.join(self.data_path, f"period_{period_id}", "visual")
            self.visual_data[period_id] = self._load_visual_data(period_visual_path)
            
            period_sensor_path = os.path.join(self.data_path, f"period_{period_id}", "sensor")
            self.sensor_data[period_id] = self._load_sensor_data(period_sensor_path)
            
            biomass_path = os.path.join(self.data_path, f"period_{period_id}", "biomass.txt")
            self.biomass[period_id] = self._load_biomass(biomass_path)
    
    def _load_visual_data(self, path):
        days = 30  
        images_per_day = 5  
        visual_data = {}
        
        for day in range(1, days+1):
            visual_data[day] = [np.random.rand(640, 640, 3) for _ in range(images_per_day)]
        
        return visual_data
    
    def _load_sensor_data(self, path):
        days = 30  
        hours_per_day = 24
        sensor_types = 4  
        sensor_data = {}
        
        for day in range(1, days+1):
            sensor_data[day] = {}
            for hour in range(1, hours_per_day+1):
                sensor_data[day][hour] = np.random.rand(sensor_types)
        
        return sensor_data
    
    def _load_biomass(self, path):
        return np.random.uniform(100, 500)
    
    def __len__(self):
        return len(self.period_list)
    
    def __getitem__(self, idx):
        period_id = self.period_list[idx]
        
        visual_data = self.visual_data[period_id]
        
        sensor_data = self.sensor_data[period_id]
        
        biomass = self.biomass[period_id]
        
        visual_tensor = self._convert_visual_to_tensor(visual_data)
        sensor_tensor = self._convert_sensor_to_tensor(sensor_data)
        biomass_tensor = torch.tensor(biomass, dtype=torch.float32)
        
        return {
            'period_id': period_id,
            'visual': visual_tensor,
            'sensor': sensor_tensor,
            'biomass': biomass_tensor
        }
    
    def _convert_visual_to_tensor(self, visual_data):
        days = len(visual_data)
        images_per_day = len(list(visual_data.values())[0])
        
        visual_tensor = torch.zeros((days, images_per_day, 3, 640, 640), dtype=torch.float32)
        
        for day, images in visual_data.items():
            for i, img in enumerate(images):
                if self.transform:
                    img = self.transform(img)
                else:
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                visual_tensor[day-1, i] = img
        
        return visual_tensor
    
    def _convert_sensor_to_tensor(self, sensor_data):
        days = len(sensor_data)
        hours_per_day = len(list(sensor_data.values())[0])
        sensor_types = len(list(sensor_data.values())[0][1])
        
        sensor_tensor = torch.zeros((days, hours_per_day, sensor_types), dtype=torch.float32)
        
        for day, hours in sensor_data.items():
            for hour, values in hours.items():
                sensor_tensor[day-1, hour-1] = torch.tensor(values, dtype=torch.float32)
        
        return sensor_tensor
