# xeda

This is a [Dagster](https://dagster.io/) project scaffolded with [`dagster project scaffold`](https://docs.dagster.io/getting-started/create-new-project).

## Prerequisites

- Python: 3.8.15

## Requirements for server:

AWS EC2 instance = 't2.large' or larger:
- RAM: 8GB+
- SSD: 16GB+

## Getting started

### Running data pipeline

a. Install data:

```bash
cd data
find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

b. Install dependencies at the root of the project:

```bash
pip install -r requirements.txt
```

c. Then, start the Docker-Compose containers responsible for running pipelines:

1. Run this command whether you want to run pipeline locally:

a. Running pipeline locally by using milvus:
```bash
docker-compose -f docker-compose.local.yaml up -d localstack milvus minio etcd
```

b. Running pipeline locally by using chromadb:
```bash
docker-compose -f docker-compose.local.yaml up -d localstack chromadb 
```

2. Run this command whether you want to deploy pipeline on server:
```bash
docker-compose -f docker-compose.deploy.yaml up -d
```

d. Then, start the Dagster UI web server:

```bash
dagster dev
```

Open http://localhost:3000 with your browser to see the UI.

### Running XEDA UI


a. Install dependencies at the root of the project:

```bash
pip install -r requirements.txt
```

b. Then, start the XEDA UI web server:

```bash
python -m web.server
```

Open http://localhost:8080 with your browser to see the UI.