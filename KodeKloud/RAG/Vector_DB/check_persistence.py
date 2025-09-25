from chromadb import PersistentClient

""" 
The check script will:

Not add anything - Will not add any new entry to the database
Reads from the persistent database - Read the entry done in the database using the save script in the previous task
 """


# Connect to the persistent DB at the specified path
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("techcorp_docs")

# Print count of documents in the collection
print("📊 Document count:", collection.count())

# Print all documents in the collection
results = collection.get()
for i, doc in enumerate(results["documents"], 1):
    print(f"{i}. {doc}")