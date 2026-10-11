"""Serve Annualyst (FastAPI app + UI) on Modal.

The container scales to zero when idle (no cost) and starts on the first request.
Vectors live in Qdrant Cloud; secrets come from the Modal secret "annualyst".

Dev (temporary URL, live reload):   uv run modal serve modal_app.py
Deploy (permanent URL):             uv run modal deploy modal_app.py
"""
import modal

INDEX_REMOTE = "/root/data/index/docling_fine5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-serve.txt")
    .env({
        "PYTHONPATH": "/root",
        "ANNUALYST_INDEX": INDEX_REMOTE,
        "ANNUALYST_STORE": "qdrant",
    })
    # code + the two data files the server needs (vectors are in Qdrant Cloud)
    .add_local_dir("rag", "/root/rag", ignore=["__pycache__", "*.ipynb", "*.pyc"])
    .add_local_dir("app", "/root/app", ignore=["__pycache__", "*.pyc"])
    .add_local_file("data/index/docling_fine5/chunks.jsonl", f"{INDEX_REMOTE}/chunks.jsonl")
    .add_local_file("data/index/docling_fine5/meta.json", f"{INDEX_REMOTE}/meta.json")
)

app = modal.App("annualyst")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("annualyst")],  # OPENAI_API_KEY, ABACI_API_KEY, QDRANT_URL, QDRANT_API_KEY
    cpu=1.0,
    memory=1024,             # MB
    min_containers=0,        # scale to zero when idle -> no cost
    scaledown_window=300,    # keep a warm container 5 min after the last request
    timeout=120,
)
@modal.concurrent(max_inputs=8)  # requests are serialized by the lock in app/main.py anyway
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app
    return fastapi_app