# Database

This folder has the database related files including the docker-compose and the schemas used for the blocking system.

Postgres was used due to the vector support via pgvector making the indexing/retrieval for blocking keys similar compared to say DuckDb where only DiskANN would be possible.

If running on Windows, the docker image should be run from within WSL to obtain better performance.

## Running

To run this enter

```
docker compose up -d
```