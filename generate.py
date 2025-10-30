import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from model import TitleGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TitleGenerator()
model.load_state_dict(torch.load('title_generator.pth', map_location=device))
model.to(device)
model.eval()

def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    next_token = top_indices[torch.multinomial(top_probs, 1)]
    return next_token.item()

def generate_title(context, max_len=10, temperature=0.8):
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding="max_length", max_length=50).to(device)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    # Start generation with first token of predicted sequence
    generated = [tokenizer.cls_token_id]
    for i in range(max_len):
        logits = outputs[0, i]
        next_token = sample_next_token(logits, temperature=temperature)
        generated.append(next_token)
        if next_token == tokenizer.sep_token_id:
            break

    title = tokenizer.decode(generated, skip_special_tokens=True)
    return title.strip().capitalize()

while True:
    context = input("\nEnter your story context (or 'q' to quit): ")
    if context.lower() == 'q':
        break
    print("\nGenerated Title:", generate_title(context))
