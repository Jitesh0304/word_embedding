from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

word_pairs = [('learn', 'learning'), ('india', 'indian'), ('fame', 'famous')]

# Compute similarity for each pair of words
for pair in word_pairs:
	tokens = tokenizer(pair, return_tensors='pt')
	with torch.no_grad():
		outputs = model(**tokens)
	
	# Extract embeddings for the [CLS] token
	cls_embedding = outputs.last_hidden_state[:, 0, :]

	similarity = torch.nn.functional.cosine_similarity(cls_embedding[0], cls_embedding[1], dim=0)
	
	print(f"Similarity between '{pair[0]}' and '{pair[1]}' using BERT: {similarity:.3f}")
