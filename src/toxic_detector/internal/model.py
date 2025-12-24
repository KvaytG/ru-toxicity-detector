import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding_dim=312):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, text_vector, dict_score):
        combined = torch.cat((text_vector, dict_score), dim=1)
        return self.net(combined)
