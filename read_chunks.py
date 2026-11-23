
import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    if r.status_code != 200:
        print("Error:", r.text)
        return None

    embedding = r.json()['embeddings']
    return embedding


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating embeddings for {json_file}")

    texts = [c['text'] for c in content['chunks']]

    embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        batch_embeddings = create_embedding(batch)

        if batch_embeddings is None:
            raise Exception(f"Embedding failed for {json_file}")

        embeddings.extend(batch_embeddings)

        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]

        chunk_id += 1
        my_dicts.append(chunk)


print("Total chunks:", len(my_dicts))

df = pd.DataFrame.from_records(my_dicts)

print(df)
