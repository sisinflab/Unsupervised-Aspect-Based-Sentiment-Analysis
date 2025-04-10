import pandas as pd
from transformers import BertTokenizer
import re
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse


from model.model import *

parser = argparse.ArgumentParser(description="Carica un file CSV")
parser.add_argument("--input_path", type=str, required=True, help="Percorso del file CSV")

args = parser.parse_args()


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("simonepretee/bert-distilbert-attentionlayers-sentiment-analysis-en")

df = pd.read_csv(args.input_path)

sentences = df["sentence"].tolist()
aspects = df["aspect"].tolist()
labels = df["label"].tolist()

model = CustomBertForSequenceClassification.from_pretrained("../model/trained_model/ABSA_model")

predictions = []

for tq_index in tqdm(range(len(sentences))):
    sentence = sentences[tq_index]
    aspect_index  = tq_index + 1
    text = re.sub(r'[^\w\s]', '', sentence).lower()
    inputs = tokenizer(text, padding="max_length", return_tensors="pt", truncation=True, max_length=512,
                       add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output_base = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        last_hidden_size = output_base[0]
        t_output = model.t_layer(last_hidden_size)

        token_classification = model.token_classifier(t_output)
        token_classification = F.softmax(token_classification, dim=2)

    subtoken_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(aspects[aspect_index]))
    indices = []
    for i in range(len(inputs['input_ids'][0]) - len(subtoken_ids) + 1):
        if (inputs['input_ids'][0][i:i + len(subtoken_ids)].cpu().numpy() == subtoken_ids).all():
            indices.extend(range(i, i + len(subtoken_ids)))

    tokens = tokenizer.tokenize(text)
    with torch.no_grad():
        output_distilbert = model.distilbert(input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = output_distilbert.attentions
        last_layer_attention = attentions[-1][0]
        avg_attention = last_layer_attention.mean(0)
        token_scores = []
        if len(tokens) > 510:
            tokens = tokens[:510]
        for i, string in enumerate(tokens):
            negative, positive = token_classification[0][i]
            score = positive - negative
            token_scores.append(score.item())
    sep_index = (input_ids[0] == 102).nonzero(as_tuple=True)[0].item()
    attention_token = avg_attention[indices].mean(axis=0)
    attention_token = attention_token[1:sep_index]
    attention_token = attention_token / attention_token.sum()
    token_scores_tensor = torch.Tensor(token_scores).to(device)
    score = sum(attention_token * token_scores_tensor)

    if score > 0:
            token_label = "positive"
    else:
            token_label = "negative"
    predictions.append(token_label.lower())

print(f"accuracy: {accuracy_score(labels, predictions)}")
print(f"F1: {f1_score(labels, predictions, average="macro")}")
print(f"precision: {precision_score(labels, predictions, average="macro")}")
print(f"recall: {recall_score(labels, predictions, average="macro")}")
