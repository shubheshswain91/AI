from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["Office equipment policy", 
        "Office furniture guidelines",
        "Office travel policy"]

analyzer = TfidfVectorizer()
word_scores = analyzer.fit_transform(docs)

print(word_scores.toarray())

query = "furniture"
query_score = analyzer.transform([query])

print(f"Query Score: {query_score.toarray()}")
print(f"Query: {query} -> Document {get_document_id(scores) + 1}")