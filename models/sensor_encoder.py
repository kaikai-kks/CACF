import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from config.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=Config.MAX_SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SensorEncoder(nn.Module):
    def __init__(self, input_dim=Config.SENSOR_INPUT_DIM, 
                 d_model=Config.TRANSFORMER_DIM, 
                 nhead=Config.TRANSFORMER_HEADS, 
                 num_layers=Config.TRANSFORMER_LAYERS,
                 dim_feedforward=Config.TRANSFORMER_FF_DIM,
                 dropout=Config.DROPOUT_RATE,
                 activation="gelu"):

        super(SensorEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_mapping = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, hours_per_day, _ = x.shape
        
        day_features = []
        
        for day_idx in range(seq_len):
            day_data = x[:, day_idx, :, :]  
            day_data = self.input_mapping(day_data)  
            
            day_data = self.pos_encoder(day_data)
            
            day_data = self.dropout(day_data)
            
            if mask is not None:
                day_mask = mask[:, day_idx] if mask.dim() > 2 else mask
                transformer_output = self.transformer_encoder(day_data, src_key_padding_mask=day_mask)
            else:
                transformer_output = self.transformer_encoder(day_data)
            
            day_feature = transformer_output.mean(dim=1)  
            
            day_feature = self.output_projection(day_feature)
            
            day_features.append(day_feature)
        
        sensor_features = torch.stack(day_features, dim=1)  
        
        return sensor_features
    
    def extract_features(self, x, mask=None):
        with torch.no_grad():
            features = self.forward(x, mask)
        return features


class SensorTransformerEncoder(SensorEncoder):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("SensorTransformerEncoder is deprecated, use SensorEncoder instead", 
                     DeprecationWarning, stacklevel=2)
        super(SensorTransformerEncoder, self).__init__(*args, **kwargs)


class SensorFeatureExtractor(nn.Module):
    def __init__(self, input_dim=Config.SENSOR_INPUT_DIM, 
                 hidden_dim=Config.FEATURE_EXTRACTOR_HIDDEN_DIM,
                 output_dim=Config.FEATURE_EXTRACTOR_OUTPUT_DIM,
                 dropout=Config.DROPOUT_RATE):
        import warnings
        warnings.warn("deprecated SensorFeatureExtractor, use SensorEncoder instead", 
                     DeprecationWarning, stacklevel=2)
        super(SensorFeatureExtractor, self).__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, input_dim)
        features = self.feature_network(x_reshaped)
        features = features.view(batch_size, seq_len, -1)
        return features


class SensorTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("deprecated SensorTransformer, use SensorEncoder instead", 
                     DeprecationWarning, stacklevel=2)
        super(SensorTransformer, self).__init__()
        self.encoder = SensorEncoder(*args, **kwargs)
        
    def forward(self, src, tgt=None, *args, **kwargs):
        return self.encoder(src)


class SensorTransformerWithFeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("deprecated SensorTransformerWithFeatureExtractor, use SensorEncoder instead", 
                     DeprecationWarning, stacklevel=2)
        super(SensorTransformerWithFeatureExtractor, self).__init__()
        self.sensor_encoder = SensorEncoder(*args, **kwargs)
        
    def forward(self, x, mask=None):
        return self.sensor_encoder(x, mask)
