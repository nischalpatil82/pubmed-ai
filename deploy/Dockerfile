FROM python:3.11-slim

# Non-root, because Hugging Face Spaces run containers as uid 1000 and anything
# the app writes (model cache, downloaded data) must be owned by that user.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PUBMED_DATASET=/home/user/data/dataset.json \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# torch is the heavy one; install deps first so this layer caches across pushes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir huggingface_hub

COPY --chown=user pipeline/ ./pipeline/
COPY --chown=user deploy/fetch_data.py .

USER user

# Spaces expect the app on 7860.
EXPOSE 7860

# fetch_data.py pulls the prebuilt store/ and index/ from the dataset repo.
# Nothing is embedded here: this selected release contains 6,290,649 vectors
# and is about 18.66 GB. Cold-start duration depends on Space storage, cache
# state and network throughput; it never recomputes embeddings on the Space.
CMD ["sh", "-c", "python fetch_data.py && python pipeline/app.py --host 0.0.0.0 --port 7860"]
