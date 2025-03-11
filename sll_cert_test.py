import requests
from huggingface_hub import configure_http_backend
from transformers import BertModel

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

if __name__ == '__main__':
    configure_http_backend(backend_factory=backend_factory)
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False, force_download=True)
    print("success maiii...")
