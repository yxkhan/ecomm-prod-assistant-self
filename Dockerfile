FROM python:3.11-slim

WORKDIR /app

# install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY prod_assistant ./prod_assistant

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# run uvicorn properly on 0.0.0.0:8000
CMD ["uvicorn", "prod_assistant.router.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]