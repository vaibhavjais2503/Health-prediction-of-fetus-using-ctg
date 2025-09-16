FROM python:3.10-slim as build

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        libpng-dev \
        libfreetype6-dev \
        libopenblas-dev \
        liblapack-dev \
        libstdc++6 \
        make \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
        libpng-dev \
        libfreetype6 \
        libopenblas-dev \
        liblapack-dev \
        libstdc++6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8501

CMD ["python3", "./src/Model_Training.py"]
CMD ["streamlit", "run", "app.py"]
