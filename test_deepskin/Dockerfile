FROM python:3.12

WORKDIR /app

RUN apt update && apt install -y libgl1-mesa-glx

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN curl -L -o Deepskin.zip https://github.com/Nico-Curti/Deepskin/archive/refs/heads/main.zip
RUN unzip Deepskin.zip
RUN mv Deepskin-main Deepskin
RUN rm -f Deepskin.zip
RUN python -m pip install --no-cache-dir -r ./Deepskin/requirements.txt
RUN python -m pip install --no-cache-dir ./Deepskin
RUN rm -rf ./Deepskin

COPY . .

ENV TF_ENABLE_ONEDNN_OPTS=0

CMD ["python", "-m", "api.main"]
