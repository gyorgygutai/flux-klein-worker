FROM runpod/base:0.4.0-cuda11.8.0

ENV HF_HOME=/runpod-volume/huggingface-cache
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY models.py .

CMD ["python3.11", "-u", "handler.py"]