FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy backend code
COPY backend/ .

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]