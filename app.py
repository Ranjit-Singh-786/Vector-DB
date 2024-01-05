from vector_store import VectorStore
import numpy as np 
vector_obj = VectorStore()

# STEPS vectorDB in memory
# 1. First make vectors from sentence
# 2. Second add the vector into vectorstore class
# 3. Find the best similar vector from our vector database

# if you will choose good technology to vector generation,
# then you will get good  results, there are many techniques for 
# vector represetation, some suggestion from my side : word2vec , Glove , Bert

sentences = [
    "I eat mango",
    "Mango is my favourite fruite",
    "Mango , apples and oranges are fruits",
    "Fruits are good for health",
    "Mukesh is richest person of asia",
    "virat kohli is great indian cricket player",
    "sachin tendulkar was great indian player",

]


# for vector representation
# step 1 : Tokenization
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)


# step 2 : Assign unique id to all the vocabulary word 
word_index = {word:i for i , word in enumerate(vocabulary)}

# step 3 : Perform vectoraization representation
sentence_vector = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary)) 
    for token in tokens:
        vector[word_index[token]] +=1
    sentence_vector[sentence] = vector

# Adding sentence and vectors into vector store
for sentence , vector in sentence_vector.items():
    vector_obj.add_vector(sentence,vector)

# Now we will check similar sentence from our vectorDB so first 
# represent query sentence as a vector
query_sentence = "virat kohli is my favourate player"
query_vector = np.zeros(len(vocabulary))
tokens = query_sentence.lower().split()
for token in tokens:
    if token in word_index:
        query_vector[word_index[token]] +=1


# Finding result from VectorDB
similar_result = vector_obj.find_similar_vector(query_vector=query_vector)
print("Query sentence :- ",query_sentence,'\n')
print(similar_result)
