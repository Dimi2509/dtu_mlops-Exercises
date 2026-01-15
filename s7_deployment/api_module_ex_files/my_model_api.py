from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up the image captioning model."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading Model")
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up Model")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.post("/model_step/")
def predict_step(
    data: UploadFile = File(...),
    max_length: int = 16,
    num_beams: int = 8,
    num_return_sequences: int = 1,
):
    """Perform image captioning on the uploaded image."""
    print(data.file)
    image = Image.open(data.file)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # Build gen_kwargs from parameters
    gen_kwargs_params = {
        "max_length": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
    }
    output_ids = model.generate(pixel_values, **gen_kwargs_params)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return {"predictions": preds, "gen_kwargs": gen_kwargs_params}
