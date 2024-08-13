FROM python:3.8.15-slim

# Change working directory
WORKDIR /usr/src/app
ENV DAGSTER_HOME=/usr/src/app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY pipeline ./pipeline

CMD ["dagit", "-h", "0.0.0.0", "-p", "3000"]
