# Database

A local Postgres 18 with the pgvector extension, defined in `docker-compose.yml`. `company_vectorize.pgvector` generates the SQL to create a vector table, index it and query it by distance; no pipeline stage connects to the database.

The user, password, database name and port (5432) are in `docker-compose.yml`. On Windows, run Docker inside WSL for better performance.

## Running

From this folder:

```
docker compose up -d
```