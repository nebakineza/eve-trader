import torch
import torch.nn as nn
import torch.nn.functional as F

class EveMarketAttention(nn.Module):
    """
    Custom Multi-Head Attention mechanism specifically tuned for EVE Online's 24-hour market cycles.
    
    EVE Online markets have distinct daily seasonality (DT - Downtime) and weekly seasonality (weekend spikes).
    This layer attends to these specific temporal features.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding specifically for 24-hour cycles (0-23 hours)
        # and 7-day cycles (0-6 days)
        self.hour_embedding = nn.Embedding(24, embed_dim)
        self.day_embedding = nn.Embedding(7, embed_dim)

    def forward(self, x, hour_indices, day_indices):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            hour_indices (torch.Tensor): Hour of day for each step (batch_size, seq_len)
            day_indices (torch.Tensor): Day of week for each step (batch_size, seq_len)
        """
        # Add cyclical temporal context to the input embedding
        temporal_context = self.hour_embedding(hour_indices) + self.day_embedding(day_indices)
        x_with_time = x + temporal_context
        
        # Self-attention to find patterns across the time window (e.g., correlation between 
        # listing prices at 18:00 EVE time vs 02:00 EVE time)
        attn_output, _ = self.mha(x_with_time, x_with_time, x_with_time)
        
        return self.layer_norm(x + attn_output)

class OracleTemporalTransformer(nn.Module):
    """
    The Oracle: A Temporal Fusion Transformer (TFT) based model for predicting 
    Next-Hour price distributions in EVE Online.
    """
    def __init__(self, input_dim, embed_dim=128, num_heads=4, forecast_horizon=1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # The core "Thinking" layer
        self.eve_attention = EveMarketAttention(embed_dim, num_heads)
        
        # LSTM for capturing immediate sequential dependencies (local context)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # Prediction heads
        self.regime_classifier = nn.Linear(embed_dim, 3) # Stable, Trending, Volatile War
        self.price_predictor = nn.Linear(embed_dim, forecast_horizon)
        self.confidence_estimator = nn.Linear(embed_dim, 1) # Probability score (0-1)

    def encode_temporal_patterns(self, market_features, hour_indices, day_indices):
        """
        Uses Multi-Head Attention to identify weekly or daily cycles.
        
        Args:
            market_features: (batch, seq, features) - Raw signals from Analyst
            hour_indices: (batch, seq) - 0-23 int
            day_indices: (batch, seq) - 0-6 int
        """
        # 1. Project to embedding space
        x = self.embedding(market_features)
        
        # 2. Apply EVE-specific attention to catch "Tuesday patch day" or "Weekend Warrior" patterns
        x_attended = self.eve_attention(x, hour_indices, day_indices)
        
        # 3. Process sequential flow
        lstm_out, _ = self.lstm(x_attended)
        
        # We take the last state for prediction
        context_vector = lstm_out[:, -1, :]
        return context_vector

    def predict_regime(self, context_vector):
        """
        Classifies the current market state.
        Returns logits for: [Stable, Trending, Volatile War]
        """
        return self.regime_classifier(context_vector)

    def confidence_scoring(self, context_vector):
        """
        Outputs a probability score.
        If score < 0.40, the Strategist should enter a 'Wait' state.
        Now normalized strictly 0-1 via Sigmoid.
        """
        return torch.sigmoid(self.confidence_estimator(context_vector))

    def forward(self, market_features, hour_indices, day_indices):
        """
        Full forward pass.
        Returns dict with predictions and flags.
        """
        context = self.encode_temporal_patterns(market_features, hour_indices, day_indices)
        
        regime_logits = self.predict_regime(context)
        predicted_price = self.price_predictor(context)
        confidence = self.confidence_scoring(context)
        
        # Check for Insufficient Data / Low Confidence
        # We process this check even though we return gradients for training.
        # In inference mode, this string flag is consumed by the Strategist.
        status_flag = "OK"
        try:
            # confidence is shape (batch, 1) during training; use mean as a stable scalar.
            conf_scalar = float(confidence.detach().mean().item())
            if conf_scalar < 0.40:
                status_flag = "INSUFFICIENT_DATA"
        except Exception:
            pass

        return {
            "predicted_price": predicted_price,
            "regime_logits": regime_logits,
            "confidence": confidence,
            "status": status_flag
        }
