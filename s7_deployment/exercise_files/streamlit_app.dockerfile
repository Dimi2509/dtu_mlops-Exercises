FROM python:3.9-slim

EXPOSE 8080

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git 

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install altair pandas streamlit

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
