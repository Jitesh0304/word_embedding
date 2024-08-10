# !pip install gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') # Download the tokenizer models if not already downloaded

sample = "Word embeddings are dense vector representations of words."
tokenized_corpus = word_tokenize(sample.lower()) # Lowercasing for consistency

skipgram_model = Word2Vec(sentences=[tokenized_corpus],
						vector_size=100, # Dimensionality of the word vectors
						window=5,		 # Maximum distance between the current and predicted word within a sentence
						sg=1,			 # Skip-Gram model (1 for Skip-Gram, 0 for CBOW)
						min_count=1,	 # Ignores all words with a total frequency lower than this
						workers=4)	 # Number of CPU cores to use for training the model

# Training
skipgram_model.train([tokenized_corpus], total_examples=1, epochs=10)
skipgram_model.save("skipgram_model.model")
loaded_model = Word2Vec.load("skipgram_model.model")
vector_representation = loaded_model.wv['word']
print("Vector representation of 'word':", vector_representation)
