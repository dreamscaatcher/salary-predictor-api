FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create model directory and run training script
RUN mkdir -p model && python train_model.py

# Set environment variables for production
ENV PORT=8000
ENV HOST=0.0.0.0

# Start the application in production mode
CMD ["uvicorn", "main:app", "--host", "${HOST}", "--port", "${PORT}", "--workers", "4"]
