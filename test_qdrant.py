from retrieval.pipeline import retrieve
print("Retrieving for student_124...")
docs = retrieve(query="what is this document about?", user_id="student_124", limit=2)
print("Docs found:", len(docs))
for d in docs:
    print(f"- {d['citation']}")
