import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from models.visual_gru import VisualGRU, YOLOFeatureExtractor
from models.sensor_encoder import SensorTransformerEncoder, SensorFeatureExtractor
from models.cross_modal_fusion import CrossModalFusion, HierarchicalCrossModalFusion


class ContextAwareAttention(nn.Module):
    
    def __init__(self, query_dim, context_dim, hidden_dim, num_heads=Config.CONTEXT_ATTENTION_HEADS, dropout=Config.DROPOUT_RATE):

        super(ContextAwareAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        self.context_modulator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, query, context, mask=None):

        batch_size, seq_len, _ = query.shape
        _, context_len, _ = context.shape
        
        Q = self.query_proj(query)  
        C = self.context_proj(context) 
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        C = C.view(batch_size, context_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        
        scores = torch.matmul(Q, C.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_vectors = torch.matmul(attention_weights, C)  
        
        context_vectors = context_vectors.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)  
        
        query_context = torch.cat([Q.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim), context_vectors], dim=-1)
        
        modulated = self.context_modulator(query_context)
        
        output = self.output_proj(modulated)  
        
        return output, attention_weights


class TemporalContextModule(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers=Config.TEMPORAL_CONTEXT_LAYERS, dropout=Config.DROPOUT_RATE):
        
        super(TemporalContextModule, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.output_mapping = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, mask=None):
        
        residual = x
        
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            
            packed_output, _ = self.lstm(packed_x)
            
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.size(1)
            )
        else:
            output, _ = self.lstm(x)
        
        output = self.dropout(output)
        
        output = self.norm(output)
        
        output = self.output_mapping(output)
        
        output = output + residual
        
        return output


class EnvironmentalContextModule(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=Config.ENV_CONTEXT_LAYERS, dropout=Config.DROPOUT_RATE):
        
        super(EnvironmentalContextModule, self).__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=Config.ENV_CONTEXT_HEADS,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.output_mapping = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, mask=None):
        
        features = self.feature_network(x)
        
        if mask is not None:
            output = self.encoder(features, src_key_padding_mask=~mask)
        else:
            output = self.encoder(features)
        
        output = self.dropout(output)
        
        output = self.output_mapping(output)
        
        return output


class CACF(nn.Module):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 env_input_dim=Config.ENV_INPUT_DIM, visual_hidden_dim=Config.GRU_HIDDEN_SIZE, 
                 sensor_hidden_dim=Config.TRANSFORMER_DIM, fusion_dim=Config.FUSION_DIM, 
                 context_dim=Config.CONTEXT_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 dropout=Config.DROPOUT_RATE):
        
        super(CACF, self).__init__()
        
        self.cross_modal_fusion = CrossModalFusion(
            visual_input_dim=visual_input_dim,
            sensor_input_dim=sensor_input_dim,
            visual_hidden_dim=visual_hidden_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            fusion_dim=fusion_dim,
            output_dim=fusion_dim, 
            num_gru_layers=num_gru_layers,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.temporal_context = TemporalContextModule(
            input_dim=fusion_dim,
            hidden_dim=context_dim,
            dropout=dropout
        )
        
        self.environmental_context = EnvironmentalContextModule(
            input_dim=env_input_dim,
            hidden_dim=context_dim,
            output_dim=context_dim,
            dropout=dropout
        )
        
        self.context_attention = ContextAwareAttention(
            query_dim=fusion_dim,
            context_dim=context_dim,
            hidden_dim=context_dim,
            dropout=dropout
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'cross_modal_fusion' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, visual_input, sensor_input, env_input, 
                visual_mask=None, sensor_mask=None, env_mask=None):
        
        fusion_features, cross_modal_attention = self.cross_modal_fusion(
            visual_input, sensor_input, visual_mask, sensor_mask
        )
        
        temporal_context = self.temporal_context(fusion_features, mask=sensor_mask)
        
        env_context = self.environmental_context(env_input, mask=env_mask)
        
        combined_context = temporal_context + env_context
        
        context_enhanced_features, context_attention = self.context_attention(
            fusion_features, combined_context, mask=sensor_mask
        )
        
        outputs = self.output_layer(context_enhanced_features)
        
        attention_weights = {
            'cross_modal': cross_modal_attention,
            'context': context_attention
        }
        
        return outputs, attention_weights
    
    def predict(self, visual_input, sensor_input, env_input, 
                visual_mask=None, sensor_mask=None, env_mask=None):
        
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(
                visual_input, sensor_input, env_input,
                visual_mask, sensor_mask, env_mask
            )
            predictions = outputs.squeeze(-1)
        return predictions


class HierarchicalCACF(nn.Module):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 env_input_dim=Config.ENV_INPUT_DIM, visual_hidden_dim=Config.GRU_HIDDEN_SIZE, 
                 sensor_hidden_dim=Config.TRANSFORMER_DIM, fusion_dim=Config.FUSION_DIM, 
                 context_dim=Config.CONTEXT_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 num_time_scales=Config.NUM_TIME_SCALES, dropout=Config.DROPOUT_RATE):
        
        super(HierarchicalCACF, self).__init__()
        
        self.hierarchical_fusion = HierarchicalCrossModalFusion(
            visual_input_dim=visual_input_dim,
            sensor_input_dim=sensor_input_dim,
            visual_hidden_dim=visual_hidden_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            fusion_dim=fusion_dim,
            output_dim=fusion_dim,  
            num_gru_layers=num_gru_layers,
            num_transformer_layers=num_transformer_layers,
            num_time_scales=num_time_scales,
            dropout=dropout
        )
        
        self.temporal_context = TemporalContextModule(
            input_dim=fusion_dim,
            hidden_dim=context_dim,
            dropout=dropout
        )
        
        self.environmental_context = EnvironmentalContextModule(
            input_dim=env_input_dim,
            hidden_dim=context_dim,
            output_dim=context_dim,
            dropout=dropout
        )
        
        self.context_attention = ContextAwareAttention(
            query_dim=fusion_dim,
            context_dim=context_dim,
            hidden_dim=context_dim,
            dropout=dropout
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'hierarchical_fusion' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, visual_input, sensor_input, env_input, 
                visual_mask=None, sensor_mask=None, env_mask=None):
        
        fusion_features, hierarchical_attention = self.hierarchical_fusion(
            visual_input, sensor_input, visual_mask, sensor_mask
        )
        
        temporal_context = self.temporal_context(fusion_features, mask=sensor_mask)
        
        env_context = self.environmental_context(env_input, mask=env_mask)
        
        combined_context = temporal_context + env_context
        
        context_enhanced_features, context_attention = self.context_attention(
            fusion_features, combined_context, mask=sensor_mask
        )
        
        outputs = self.output_layer(context_enhanced_features)
        
        attention_weights = {
            'hierarchical': hierarchical_attention,
            'context': context_attention
        }
        
        return outputs, attention_weights
    
    def predict(self, visual_input, sensor_input, env_input, 
                visual_mask=None, sensor_mask=None, env_mask=None):
        
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(
                visual_input, sensor_input, env_input,
                visual_mask, sensor_mask, env_mask
            )
            predictions = outputs.squeeze(-1)
        return predictions


class AdaptiveCACF(nn.Module):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 env_input_dim=Config.ENV_INPUT_DIM, visual_hidden_dim=Config.GRU_HIDDEN_SIZE, 
                 sensor_hidden_dim=Config.TRANSFORMER_DIM, fusion_dim=Config.FUSION_DIM, 
                 context_dim=Config.CONTEXT_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 dropout=Config.DROPOUT_RATE):
        
        super(AdaptiveCACF, self).__init__()
        
        self.cacf = CACF(
            visual_input_dim=visual_input_dim,
            sensor_input_dim=sensor_input_dim,
            env_input_dim=env_input_dim,
            visual_hidden_dim=visual_hidden_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            fusion_dim=fusion_dim,
            context_dim=context_dim,
            output_dim=output_dim,
            num_gru_layers=num_gru_layers,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.visual_quality_estimator = nn.Sequential(
            nn.Linear(visual_input_dim, visual_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.sensor_quality_estimator = nn.Sequential(
            nn.Linear(sensor_input_dim, sensor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sensor_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'cacf' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, visual_input, sensor_input, env_input, 
                visual_mask=None, sensor_mask=None, env_mask=None):
        
        batch_size, seq_len = visual_input.shape[0], visual_input.shape[1]
        visual_features = visual_input.view(batch_size, seq_len, -1)
        visual_quality = self.visual_quality_estimator(visual_features)  
        
        sensor_quality = self.sensor_quality_estimator(sensor_input)  
        
        outputs, attention_weights = self.cacf(
            visual_input, sensor_input, env_input,
            visual_mask, sensor_mask, env_mask
        )
        
