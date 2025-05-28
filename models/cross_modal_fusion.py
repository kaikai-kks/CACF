import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from models.visual_gru import VisualGRU, YOLOFeatureExtractor
from models.sensor_encoder import SensorTransformerEncoder, SensorFeatureExtractor


class CrossAttention(nn.Module):
    
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, num_heads=Config.CROSS_ATTENTION_HEADS, dropout=Config.DROPOUT_RATE):

        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, query, key, value, mask=None):

        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape
        
        Q = self.query_proj(query)  
        K = self.key_proj(key)      
        V = self.value_proj(value)  
        
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)  
        
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.hidden_dim)  
        
        output = self.output_proj(context)  
        
        return output, attention_weights


class ModalityFusion(nn.Module):
    
    def __init__(self, visual_dim, sensor_dim, fusion_dim, dropout=Config.DROPOUT_RATE):
        
        super(ModalityFusion, self).__init__()
        
        self.visual_mapping = nn.Linear(visual_dim, fusion_dim)
        self.sensor_mapping = nn.Linear(sensor_dim, fusion_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.visual_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.sensor_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, visual_features, sensor_features):
        
        visual_mapped = self.visual_mapping(visual_features)
        sensor_mapped = self.sensor_mapping(sensor_features)
        
        concat_features = torch.cat([visual_mapped, sensor_mapped], dim=-1)
        
        visual_gate_weights = torch.sigmoid(self.visual_gate(concat_features))
        sensor_gate_weights = torch.sigmoid(self.sensor_gate(concat_features))
        
        gated_visual = visual_mapped * visual_gate_weights
        gated_sensor = sensor_mapped * sensor_gate_weights
        
        fused_features = self.fusion_layer(torch.cat([gated_visual, gated_sensor], dim=-1))
        
        fused_features = self.dropout(fused_features)
        
        return fused_features


class CrossModalFusion(nn.Module):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 visual_hidden_dim=Config.GRU_HIDDEN_SIZE, sensor_hidden_dim=Config.TRANSFORMER_DIM,
                 fusion_dim=Config.FUSION_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 dropout=Config.DROPOUT_RATE):
        
        super(CrossModalFusion, self).__init__()
        
        self.visual_feature_extractor = YOLOFeatureExtractor()
        
        self.sensor_feature_extractor = SensorFeatureExtractor(
            input_dim=sensor_input_dim,
            output_dim=sensor_input_dim  
        )
        
        self.visual_gru = VisualGRU(
            input_size=visual_input_dim,
            hidden_size=visual_hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
            feature_extractor=self.visual_feature_extractor
        )
        
        self.sensor_transformer = SensorTransformerEncoder(
            input_dim=sensor_input_dim,
            d_model=sensor_hidden_dim,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.visual_to_sensor_attention = CrossAttention(
            query_dim=visual_hidden_dim,
            key_dim=sensor_hidden_dim,
            value_dim=sensor_hidden_dim,
            hidden_dim=fusion_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        self.sensor_to_visual_attention = CrossAttention(
            query_dim=sensor_hidden_dim,
            key_dim=visual_hidden_dim,
            value_dim=visual_hidden_dim,
            hidden_dim=fusion_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        self.modality_fusion = ModalityFusion(
            visual_dim=fusion_dim,
            sensor_dim=fusion_dim,
            fusion_dim=fusion_dim,
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
            if 'visual_feature_extractor' not in name and 'sensor_feature_extractor' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, visual_input, sensor_input, visual_mask=None, sensor_mask=None):
        
        visual_features, _ = self.visual_gru(visual_input)  
        
        sensor_features = self.sensor_transformer(sensor_input, mask=sensor_mask) 
        
        enhanced_visual, v2s_attention = self.visual_to_sensor_attention(
            visual_features, sensor_features, sensor_features, mask=sensor_mask
        )  
        
        enhanced_sensor, s2v_attention = self.sensor_to_visual_attention(
            sensor_features, visual_features, visual_features, mask=visual_mask
        )  
        
        fused_features = self.modality_fusion(enhanced_visual, enhanced_sensor)  
        
        outputs = self.output_layer(fused_features)  
        
        attention_weights = {
            'visual_to_sensor': v2s_attention,
            'sensor_to_visual': s2v_attention
        }
        
        return outputs, attention_weights
    
    def predict(self, visual_input, sensor_input, visual_mask=None, sensor_mask=None):
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(visual_input, sensor_input, visual_mask, sensor_mask)
            predictions = outputs.squeeze(-1)
        return predictions


class CrossModalFusionWithTemporalAttention(CrossModalFusion):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 visual_hidden_dim=Config.GRU_HIDDEN_SIZE, sensor_hidden_dim=Config.TRANSFORMER_DIM,
                 fusion_dim=Config.FUSION_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 dropout=Config.DROPOUT_RATE):

        super(CrossModalFusionWithTemporalAttention, self).__init__(
            visual_input_dim, sensor_input_dim, visual_hidden_dim, sensor_hidden_dim,
            fusion_dim, output_dim, num_gru_layers, num_transformer_layers, dropout
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=Config.TEMPORAL_ATTENTION_HEADS,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
    
    def forward(self, visual_input, sensor_input, visual_mask=None, sensor_mask=None):

        fused_features, attention_weights = super().forward(visual_input, sensor_input, visual_mask, sensor_mask)
        batch_size, seq_len, _ = fused_features.shape
        
        attn_mask = None
        if sensor_mask is not None:
            attn_mask = ~sensor_mask.bool()
        
        fused_norm = self.norm1(fused_features)
        
        temporal_output, temporal_attn_weights = self.temporal_attention(
            fused_norm, fused_norm, fused_norm,
            key_padding_mask=attn_mask
        )
        
        fused_features = fused_features + temporal_output
        
        fused_norm = self.norm2(fused_features)
        
        ff_output = self.feed_forward(fused_norm)
        
        fused_features = fused_features + ff_output
        
        outputs = self.output_layer(fused_features)
        
        attention_weights['temporal'] = temporal_attn_weights
        
        return outputs, attention_weights


class HierarchicalCrossModalFusion(nn.Module):
    
    def __init__(self, visual_input_dim=3, sensor_input_dim=Config.SENSOR_INPUT_DIM, 
                 visual_hidden_dim=Config.GRU_HIDDEN_SIZE, sensor_hidden_dim=Config.TRANSFORMER_DIM,
                 fusion_dim=Config.FUSION_DIM, output_dim=1,
                 num_gru_layers=Config.GRU_NUM_LAYERS, num_transformer_layers=Config.TRANSFORMER_LAYERS,
                 num_time_scales=Config.NUM_TIME_SCALES, dropout=Config.DROPOUT_RATE):
        
        super(HierarchicalCrossModalFusion, self).__init__()
        
        self.num_time_scales = num_time_scales
        
        self.visual_feature_extractor = YOLOFeatureExtractor()
        
        self.sensor_feature_extractor = SensorFeatureExtractor(
            input_dim=sensor_input_dim,
            output_dim=sensor_input_dim
        )
        
        self.visual_grus = nn.ModuleList([
            VisualGRU(
                input_size=visual_input_dim,
                hidden_size=visual_hidden_dim,
                num_layers=num_gru_layers,
                dropout=dropout,
                feature_extractor=None  
            ) for _ in range(num_time_scales)
        ])
        
        self.sensor_transformers = nn.ModuleList([
            SensorTransformerEncoder(
                input_dim=sensor_input_dim,
                d_model=sensor_hidden_dim,
                num_layers=num_transformer_layers,
                dropout=dropout
            ) for _ in range(num_time_scales)
        ])
        
        self.cross_modal_fusions = nn.ModuleList([
            CrossModalFusionWithTemporalAttention(
                visual_input_dim=visual_hidden_dim,  
                sensor_input_dim=sensor_hidden_dim,  
                visual_hidden_dim=visual_hidden_dim,
                sensor_hidden_dim=sensor_hidden_dim,
                fusion_dim=fusion_dim,
                output_dim=fusion_dim,  
                num_gru_layers=1,  
                num_transformer_layers=1,  
                dropout=dropout
            ) for _ in range(num_time_scales)
        ])
        
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=Config.SCALE_ATTENTION_HEADS,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'visual_feature_extractor' not in name and 'sensor_feature_extractor' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def _downsample(self, x, scale):
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        new_seq_len = seq_len // scale
        if new_seq_len * scale < seq_len:
            x = x[:, :new_seq_len * scale]
        
        if scale > 1:
            new_shape = [batch_size, new_seq_len, scale] + list(x.shape[2:])
            x_reshaped = x.reshape(*new_shape)
            
            x_downsampled = x_reshaped.mean(dim=2)
        else:
            x_downsampled = x
        
        return x_downsampled
    
    def forward(self, visual_input, sensor_input, visual_mask=None, sensor_mask=None):
        
        batch_size, seq_len = visual_input.shape[0], visual_input.shape[1]
        
        visual_features = self.visual_feature_extractor(visual_input) 
        sensor_features = self.sensor_feature_extractor(sensor_input)  
        
        scale_features = []
        scale_attention_weights = []
        
        for i in range(self.num_time_scales):
            scale = 2 ** i  # 1, 2, 4, ...
            
            visual_scale = self._downsample(visual_features, scale)
            sensor_scale = self._downsample(sensor_input, scale)
            
            visual_mask_scale = None
            sensor_mask_scale = None
            if visual_mask is not None:
                visual_mask_scale = self._downsample(visual_mask.float(), scale).bool()
            if sensor_mask is not None:
                sensor_mask_scale = self._downsample(sensor_mask.float(), scale).bool()
            
            visual_output, _ = self.visual_grus[i](visual_scale)
            sensor_output = self.sensor_transformers[i](sensor_scale, mask=sensor_mask_scale)
            
            fusion_output, attn_weights = self.cross_modal_fusions[i](
                visual_output, sensor_output, 
                visual_mask=visual_mask_scale, 
                sensor_mask=sensor_mask_scale
            )
            
            if scale > 1:
                fusion_output = fusion_output.repeat_interleave(scale, dim=1)
                fusion_output = fusion_output[:, :seq_len]
            
            scale_features.append(fusion_output)
            scale_attention_weights.append(attn_weights)
        
        stacked_features = torch.stack(scale_features, dim=1) 
        stacked_features = stacked_features.transpose(1, 2)  
        
        original_shape = stacked_features.shape
        stacked_features = stacked_features.reshape(batch_size * seq_len, self.num_time_scales, -1)  # [batch_size*seq_len, num_scales, fusion_dim]
        
        scale_query = self.norm(stacked_features)
        scale_output, scale_attn = self.scale_attention(
            scale_query, scale_query, scale_query
        )
        
        stacked_features = stacked_features + scale_output
        
        fused_features = torch.sum(stacked_features * F.softmax(scale_attn, dim=-1).transpose(-2, -1), dim=1)  # [batch_size*seq_len, fusion_dim]
        
        fused_features = fused_features.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, fusion_dim]
        
        outputs = self.output_layer(fused_features)  # [batch_size, seq_len, output_dim]
        
        attention_weights = {
            'scale_attention': scale_attn,
            'modal_attentions': scale_attention_weights
        }
        
        return outputs, attention_weights
    
    def predict(self, visual_input, sensor_input, visual_mask=None, sensor_mask=None):
        self.eval()
