import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embedding_dim=312):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, text_vector, dict_score):
        if text_vector.dim() == 1:
            text_vector = text_vector.unsqueeze(0)
        if dict_score.dim() == 1:
            dict_score = dict_score.unsqueeze(0)
        combined = torch.cat((text_vector, dict_score), dim=1)
        return self.net(combined)
