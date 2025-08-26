import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextBranch(nn.Module):
    """HF encoder (DistilBERT by default) → Linear to 64-D features."""
    def __init__(self, model_name="distilbert-base-uncased", out_dim=64, freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone  = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.backbone.config, "hidden_size", 768)
        self.proj = nn.Linear(hidden, out_dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def encode_texts(self, texts, device="cpu", max_len=128):
        toks = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {k: v.to(device) for k, v in toks.items()}

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)  # (B, out_dim)
