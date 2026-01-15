FROM nvcr.io/nvidia/pytorch:25.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install --no-cache-dir fastapi uvicorn transformers Pillow python-multipart
COPY ./ /code/app

CMD ["uvicorn", "app.my_model_api:app", "--host", "0.0.0.0", "--port", "80"]