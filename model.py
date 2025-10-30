import torch
import torch.nn as nn
from transformers import DistilBertModel

class TitleGenerator(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1):
        super(TitleGenerator, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        self.lstm = nn.LSTM(
            input_size=self.bert.config.dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_outputs.last_hidden_state)
        out = self.fc(lstm_out)
        return out
