import os
import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from typing import Optional
from urllib import response

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


# Start and shutdown events can be added if needed
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Loading ML models...")  # Startup actions
    yield
    print("Cleaning up ML models...")  # Shutdown actions


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/health")
def health_check():
    return {"message": HTTPStatus.OK.phrase, "status_code": HTTPStatus.OK}


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open(f"{FILE_PATH}/database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.get("/text_model/")
def contains_email(data: str):
    """Simple function to check if an email is valid."""
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }


class DomainEnum(Enum):
    """Domain enum."""

    gmail = "gmail"
    hotmail = "hotmail"


class Item(BaseModel):
    """Item model."""

    email: str
    domain: DomainEnum


@app.post("/text_model_domain/")
def contains_email_domain(data: Item):
    """Simple function to check if an email is valid."""
    if data.domain is DomainEnum.gmail:
        regex = r"\b[A-Za-z0-9._%+-]+@gmail+\.[A-Z|a-z]{2,}\b"
    if data.domain is DomainEnum.hotmail:
        regex = r"\b[A-Za-z0-9._%+-]+@hotmail+\.[A-Z|a-z]{2,}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None,
    }


@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...), h: Optional[int] = 224, w: Optional[int] = 224
):
    with open(f"{FILE_PATH}/image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread(f"{FILE_PATH}/image.jpg")
    res = cv2.resize(img, (h, w))
    cv2.imwrite(f"{FILE_PATH}/resized_image.jpg", res)

    response = {
        "input": data,
        "output": FileResponse(f"{FILE_PATH}/resized_image.jpg"),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
