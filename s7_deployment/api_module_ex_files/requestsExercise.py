import os

import requests

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
LOCAL_HOST = "http://localhost:8000"
# response = requests.get("https://imgs.xkcd.com/comics/making_progress.png")

# with open(os.path.join(FILE_PATH, 'img.png'),'wb') as f:
#     f.write(response.content)
# print("Image downloaded and saved as img.png")

# pload = {"username": "Olivia", "password": "123"}
# response = requests.post("https://httpbin.org/post", data=pload)
# print(response.text)

# response = requests.get(
#     LOCAL_HOST + "/query_items",
#     params={"item_id": 14},
# )

# response = requests.post(
#     LOCAL_HOST + "/login/",
#     params={"username": "Olivia", "password": "123"},
# )
# print(response.text)

# response = requests.get(
#     LOCAL_HOST + "/text_model/",
#     params={"data": "ivandm2509@gmail.com"},
# )

# print(f"Response for email check: {response.json()}")

# data = {"email": "user@gmail.com", "domain": "gmail"}
# response = requests.post(LOCAL_HOST + "/text_model_domain/", json=data)
# print(response.json())

# url = "http://localhost:8000/cv_model/"
# files = {"data": open(f"{FILE_PATH}/boat.webp", "rb")}
# params = {"h": 128, "w": 128}
# response = requests.post(url, files=files, params=params)
# print(response.json())

url = "http://localhost/model_step/"
files = {"data": open(f"{FILE_PATH}/img.png", "rb")}
params = {"max_length": 24, "num_beams": 16, "num_return_sequences": 2}
response = requests.post(url, files=files, params=params)

print(response.json())
