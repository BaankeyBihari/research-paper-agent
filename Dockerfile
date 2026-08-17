FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py simulator.py compare_models.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

VOLUME ["/data/papers"]

EXPOSE 5001

ENTRYPOINT ["./entrypoint.sh"]
