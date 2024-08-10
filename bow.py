from sklearn.feature_extraction.text import CountVectorizer

documents = ["This is the first document.",
			"This document is the second document.",
			"And this is the third one.",
			"Is this the first document?"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("Bag-of-Words Matrix:")
print(X.toarray())
print("Vocabulary (Feature Names):", feature_names)
