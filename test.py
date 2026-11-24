import requests

r = requests.post("http://localhost:11434/api/embed", json={
    "model": "bge-m3",
    "input": ["Hello world"]
})

print(r.status_code)
print(r.json().keys())
print(len(r.json()['embeddings']))