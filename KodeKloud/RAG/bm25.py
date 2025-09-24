from rank_bm25 import BM25Okapi

docs = ["Office equipment policy", 
        "Office furniture guidelines",
        "Office travel policy"]

bm25 = BM25Okapi([doc.split(" ") for doc in docs])

query = "furniture"
query_scores = bm25.get_scores(query.split(" "))
print(f"Query scores: {query_scores}")

      