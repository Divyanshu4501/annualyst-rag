FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies first (cached between builds unless requirements change)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# only the code the server needs
COPY rag/ rag/
COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]