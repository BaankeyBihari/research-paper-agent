FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py simulator.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

VOLUME ["/data/papers"]

EXPOSE 5001

ENTRYPOINT ["./entrypoint.sh"]
