import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores  = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (x * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class AegisLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden1=128, hidden2=64,
                 fc_dim=128, horizon=24, n_targets=2, dropout=0.2):
        super().__init__()

        self.bilstm1 = nn.LSTM(
            input_size=input_dim, hidden_size=hidden1,
            batch_first=True, bidirectional=True, dropout=0.0
        )
        self.drop1 = nn.Dropout(dropout)

        self.bilstm2 = nn.LSTM(
            input_size=hidden1 * 2, hidden_size=hidden2,
            batch_first=True, bidirectional=True, dropout=0.0
        )
        self.drop2 = nn.Dropout(dropout)

        self.attention = AdditiveAttention(hidden_dim=hidden2 * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden2 * 2, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.output = nn.Linear(fc_dim, horizon * n_targets)
        self.horizon   = horizon
        self.n_targets = n_targets

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out = self.drop1(out)
        out, _ = self.bilstm2(out)
        out = self.drop2(out)
        context, _ = self.attention(out)
        out = self.fc(context)
        out = self.output(out)
        return out.view(-1, self.horizon, self.n_targets)