FROM python:3.11-slim

# Non-root, because Hugging Face Spaces run containers as uid 1000 and anything
# the app writes (model cache, downloaded data) must be owned by that user.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PUBMED_STORE=/home/user/data/store \
    PUBMED_INDEX=/home/user/data/index \
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
# Nothing is embedded here - the 238,224 vectors were computed once and travel
# as files. A cold start is a ~931 MB download, not 15 hours of GPU-less work.
CMD ["sh", "-c", "python fetch_data.py && python pipeline/app.py --host 0.0.0.0 --port 7860"]
