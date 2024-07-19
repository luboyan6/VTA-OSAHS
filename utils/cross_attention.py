import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from feature_selector import FeatureSelector

class CrossAttention(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)


    def forward(self, query, key, value):

        query, key, value = query.to("cuda"), key.to("cuda"), value.to("cuda")

        query = query.to(torch.float32)
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v)

        feature_selector_q = query

        fea_sec_model = FeatureSelector(feature_selector_q.shape[2]).to("cuda")

        fea_sec_embedding = fea_sec_model(feature_selector_q)

        fea_sec_embedding = torch.nn.functional.normalize(fea_sec_embedding, p=2, dim=0)

        attention_output_norm = self.norm(attention_output + fea_sec_embedding)

        output = self.out_linear(attention_output_norm)

        return output








