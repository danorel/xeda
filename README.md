# xeda

This is a [Dagster](https://dagster.io/) project scaffolded with [`dagster project scaffold`](https://docs.dagster.io/getting-started/create-new-project).

## Getting started

a. Install data:

```bash
cd data
find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

b. Install dependencies:

```bash
pip install -r requirements.txt
```

c. Then, start the Docker-Compose containers responsible for pipelines:

```bash
docker-compose up -d
```

d. Then, start the Dagster UI web server:

```bash
dagster dev
```

Open http://localhost:3000 with your browser to see the project.
