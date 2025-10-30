import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

def load_and_prepare_data(path='dataset.csv', max_len=50):
    df = pd.read_csv(path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(
        df['context'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    labels = tokenizer(
        df['title'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )

    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    train_data = {key: val[train_idx] for key, val in inputs.items()}
    val_data = {key: val[val_idx] for key, val in inputs.items()}
    train_labels = labels['input_ids'][train_idx]
    val_labels = labels['input_ids'][val_idx]
    return train_data, train_labels, val_data, val_labels, tokenizer
