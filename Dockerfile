FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY env /app/env
COPY tasks /app/tasks
COPY server /app/server
COPY inference.py /app/
COPY openenv.yaml /app/

# OpenEnv HTTP port requirements / HF spaces default
EXPOSE 7860
ENV PORT=7860

# Command to run the application using uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
