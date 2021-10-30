# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-mpnet-base-v2') #all-mpnet-base-v2  #all-MiniLM-L6-v2
# # model = SentenceTransformer('all-MiniLM-L6-v2') #all-mpnet-base-v2  #all-MiniLM-L6-v2

# query_embedding = model.encode(['🐶', '🐱', '😍', '😄', 'dog', 'cat', 'love', 'happy'])#, 'That is a happy dog', 'That is a happy 🐶'])
# passage_embedding = model.encode(['🐶', '🐱', '😍', '😄', 'dog', 'cat', 'love', 'happy'])  #, 'That is a happy dog', 'That is a happy 🐶',
#                                 #   'That is a happy person',
#                                 #   'Today is a sunny day'])

# print("Similarity:", util.cos_sim(query_embedding, passage_embedding))


# 'That is a happy dog' 'That is a happy 🐶' 0.7239
# 'That is a happy dog' 'That is a happy person' 0.7640

# 'That is a happy 🐶' 'That is a happy dog' 0.7239
# 'That is a happy 🐶' 'That is a happy person' 0.7160


# ['🐶', '🐱', '😍', '😄']
# ['dog', 'cat', 'love', 'happy']
# [[0.3345, 0.2396, 0.3513, 0.3519],
# [0.3345, 0.2396, 0.3513, 0.3519],
# [0.3345, 0.2396, 0.3513, 0.3519],
# [0.3345, 0.2396, 0.3513, 0.3519]


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted', 'is🐶?🐱?😍?😄', '🐱', '😍', '😄']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens') #all-distilroberta-v1
# model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences)#, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)
tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids[2])
print(tokens)


# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
# print(tokens)
# # output = model(encoded_input)