FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV GPU_DISPATCHER_HOST=0.0.0.0
ENV GPU_DISPATCHER_PORT=11435
ENV GPU_DISPATCHER_DATA=/data/dispatcher.db
EXPOSE 11435

CMD ["python", "app.py"]
