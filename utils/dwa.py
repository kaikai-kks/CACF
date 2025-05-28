import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config


class DynamicWeightAllocator(nn.Module):
    
    def __init__(self, feature_dim, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(DynamicWeightAllocator, self).__init__()
        
        self.quality_estimator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.importance_estimator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, features):
        
        quality_scores = self.quality_estimator(features)  
        
        importance_scores = self.importance_estimator(features)  
        
        combined_scores = quality_scores * importance_scores  
        
        weights = combined_scores / (combined_scores.sum(dim=1, keepdim=True) + 1e-8)  
        
        return weights, quality_scores, importance_scores


class ModalityDWA(nn.Module):
    
    def __init__(self, modality_dims, fusion_dim=Config.FUSION_DIM, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(ModalityDWA, self).__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        
        self.modality_mappings = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modality_dims.items()
        })
        
        self.weight_allocator = DynamicWeightAllocator(
            feature_dim=fusion_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, modality_features):
        
        batch_size = next(iter(modality_features.values())).shape[0]
        
        mapped_features = {}
        for modality in self.modalities:
            if modality in modality_features:
                mapped_features[modality] = self.modality_mappings[modality](modality_features[modality])
        
        stacked_features = torch.stack([mapped_features[modality] for modality in self.modalities if modality in mapped_features], dim=1)  
        
        weights, quality_scores, importance_scores = self.weight_allocator(stacked_features)  
        weighted_features = stacked_features * weights  
        fused_features = weighted_features.sum(dim=1)  
        
        fused_features = self.fusion_layer(fused_features)  
        
        return fused_features, weights, quality_scores, importance_scores


class FeatureDWA(nn.Module):
    
    def __init__(self, feature_dim, num_features, output_dim=None, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(FeatureDWA, self).__init__()
        
        if output_dim is None:
            output_dim = feature_dim
        
        self.weight_allocator = DynamicWeightAllocator(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.output_mapping = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, features):
        
        weights, quality_scores, importance_scores = self.weight_allocator(features)  
        
        weighted_features = features * weights  
        
        fused_features = weighted_features.sum(dim=1)  
        
        output = self.output_mapping(fused_features)  
        
        return output, weights, quality_scores, importance_scores


class HierarchicalDWA(nn.Module):
    
    def __init__(self, modality_dims, modality_feature_counts, fusion_dim=Config.FUSION_DIM, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(HierarchicalDWA, self).__init__()
        
        self.modality_dims = modality_dims
        self.modality_feature_counts = modality_feature_counts
        self.modalities = list(modality_dims.keys())
        
        self.feature_dwas = nn.ModuleDict({
            modality: FeatureDWA(
                feature_dim=dim,
                num_features=modality_feature_counts[modality],
                output_dim=fusion_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for modality, dim in modality_dims.items()
        })
        
        self.modality_dwa = ModalityDWA(
            modality_dims={modality: fusion_dim for modality in modality_dims},
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'feature_dwas' not in name and 'modality_dwa' not in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name and 'feature_dwas' not in name and 'modality_dwa' not in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, modality_features):
        
        feature_fused = {}
        feature_weights = {}
        feature_quality = {}
        feature_importance = {}
        
        for modality in self.modalities:
            if modality in modality_features:
                feature_fused[modality], feature_weights[modality], feature_quality[modality], feature_importance[modality] = self.feature_dwas[modality](modality_features[modality])
        
        fused_features, modality_weights, modality_quality, modality_importance = self.modality_dwa(feature_fused)
        
        return fused_features, feature_weights, modality_weights, feature_quality, feature_importance, modality_quality, modality_importance


class TemporalDWA(nn.Module):
    
    def __init__(self, feature_dim, context_window=Config.DWA_CONTEXT_WINDOW, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(TemporalDWA, self).__init__()
        
        self.context_window = context_window
        
        self.context_encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.weight_allocator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.output_mapping = nn.Linear(feature_dim, feature_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'context_encoder' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, features, mask=None):
        
        batch_size, seq_len, feature_dim = features.shape
        
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            
            packed_context, _ = self.context_encoder(packed_features)
            
            context, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True, total_length=seq_len
            )
        else:
            context, _ = self.context_encoder(features)
        
        weights = self.weight_allocator(context)  
        
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()
        
        weights_sum = weights.sum(dim=1, keepdim=True)
        normalized_weights = weights / (weights_sum + 1e-8)  
        
        weighted_features = features * normalized_weights  
        
        weighted_features = self.output_mapping(weighted_features)  
        
        return weighted_features, normalized_weights


class AdaptiveDWA(nn.Module):
    
    def __init__(self, modality_dims, fusion_dim=Config.FUSION_DIM, hidden_dim=Config.DWA_HIDDEN_DIM, dropout=Config.DROPOUT_RATE):
        
        super(AdaptiveDWA, self).__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        
        self.modality_mappings = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modality_dims.items()
        })
        
        self.weight_allocator = DynamicWeightAllocator(
            feature_dim=fusion_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.strategy_network = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, modality_features):
        
        batch_size = next(iter(modality_features.values())).shape[0]
        
        mapped_features = {}
        for modality in self.modalities:
            if modality in modality_features:
                mapped_features[modality] = self.modality_mappings[modality](modality_features[modality])
        
        stacked_features = torch.stack([mapped_features[modality] for modality in self.modalities if modality in mapped_features], dim=1)  
        
        weights, quality_scores, importance_scores = self.weight_allocator(stacked_features)  
        
        concat_features = torch.cat([mapped_features[modality] for modality in self.modalities if modality in mapped_features], dim=1)  
        
        strategy_weights = F.softmax(self.strategy_network(concat_features), dim=1)  
        
        avg_weights = torch.ones_like(weights) / weights.size(1)
        
        quality_weights = quality_scores / (quality_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        importance_weights = importance_scores / (importance_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        combined_weights = (
            strategy_weights[:, 0:1].unsqueeze(1) * avg_weights +
            strategy_weights[:, 1:2].unsqueeze(1) * quality_weights +
            strategy_weights[:, 2:3].unsqueeze(1) * importance_weights
        )  
        
        weighted_features = stacked_features * combined_weights  
        
        fused_features = weighted_features.sum(dim=1)  
        
        fused_features = self.fusion_layer(fused_features)  
        
        return fused_features, combined_weights, quality_scores, importance_scores, strategy_weights
