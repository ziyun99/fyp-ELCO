from transformers import AutoTokenizer, AutoModel

# Sentences we want sentence embeddings for
sentences = ['[EM]money[ENG] [EMO]wings[EM][EM]worker[EM]][QRY]][DOC] [KIK]ðð¶', 'hiðð¶ woman in tuxedo mirror  ð±?ð ', 'This is an example sentence', 'is ð¶?ð±?ð?  ð', 'ð±', 'ð', 'ð']

# Load model from HuggingFace Hub
model_name = ["xlm-roberta-base", 'cardiffnlp/twitter-roberta-base-emoji', 'sentence-transformers/all-distilroberta-v1', 'sentence-transformers/all-mpnet-base-v2', 'sentence-transformers/bert-base-nli-mean-tokens', 'sentence-transformers/all-MiniLM-L6-v2', "sentence-transformers/distiluse-base-multilingual-cased-v2"]
# model_idx = 1
# tokenizer = AutoTokenizer.from_pretrained(model_name[model_idx])
# model = AutoModel.from_pretrained(model_name[1])

# Tokenize sentences
# encoded_input = tokenizer(sentences)#, padding=True, truncation=True, return_tensors='pt')
# print(encoded_input)
# tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids[2])
# print(tokens)

for model in model_name:
    print('\n', model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.add_tokens(['ð¶'], special_tokens=True)
    decoded_input = tokenizer.decode(tokenizer.encode(sentences[0]))
    print(decoded_input)

    encoded_input = tokenizer(sentences[0])#, padding=True, truncation=True, return_tensors='pt')
    # print(encoded_input)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    print(tokens)

    id1 = tokenizer.convert_tokens_to_ids('[EM]z')
    id2 = tokenizer.convert_tokens_to_ids('ðð¶')
    id3 = tokenizer.convert_tokens_to_ids('ð¶')
    id4 = tokenizer.convert_tokens_to_ids('ð±?ð')
    print(len(tokenizer), id1, id2, id3, id4)


from sentence_transformers import SentenceTransformer, models
tokens = ["[DOC]", "[QRY]", "[EM]", "[KIK]"]

# model = SentenceTransformer('all-MiniLM-L6-v2')
# word_embedding_model = model._first_module()
# word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
# word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
# tokenizer = word_embedding_model.tokenizer

for model in model_name:
    print('\n', model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(len(tokenizer))
    tokenizer.add_tokens(tokens, special_tokens=True)
    # print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    print(tokenizer.all_special_ids, len(tokenizer))
    decoded_input = tokenizer.decode(tokenizer.encode(sentences[0]))
    print(decoded_input)

    encoded_input = tokenizer(sentences[0])#, padding=True, truncation=True, return_tensors='pt')
    # print(encoded_input)
    convert_tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    print(convert_tokens)

    id1 = tokenizer.convert_tokens_to_ids('[EM]z')
    id2 = tokenizer.convert_tokens_to_ids('ðð¶')
    id3 = tokenizer.convert_tokens_to_ids('ð¶')
    id4 = tokenizer.convert_tokens_to_ids('ð±?ð')
    print(len(tokenizer), id1, id2, id3, id4)







# from sentence_transformers import SentenceTransformer, util, models

# #Our sentences we like to encode
# sentences = ['ð', 'ðð', 'ð¶', 'dog', 'ð']

# word_embedding_model = models.Transformer('cardiffnlp/twitter-roberta-base-emoji')
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# roberta_emoji_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# model_name = ["xlm-roberta-base", 'all-distilroberta-v1', 'all-mpnet-base-v2', 'bert-base-nli-mean-tokens', 'all-MiniLM-L6-v2']
# for m in model_name:
#     print(m)
#     model = SentenceTransformer(m) if type(m) == str else m
#     #Sentences are encoded by calling model.encode()
#     sentence_embeddings = model.encode(sentences)

#     #Print the embeddings
#     for sentence, embedding in zip(sentences, sentence_embeddings):
#         print("Sentence:", sentence)
#         print("Embedding:", embedding[:10])
#         print("")                                                                                 
# #     query_embedding = sentence_model.encode('That is a happy ð¶')
# #     passage_embedding = sentence_model.encode('That is a happy dog')                 
# #     print("Similarity:", util.cos_sim(query_embedding, passage_embedding))




