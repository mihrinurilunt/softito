# Docker Document Processor

A beginner-friendly, production-inspired mini project that demonstrates a FastAPI backend running with PostgreSQL and pgAdmin through Docker Compose.

Users can upload `.txt` files. The API saves each file, analyzes the text, stores metadata in PostgreSQL, and returns a clean JSON response.

## Architecture

```text
+-------------+          HTTP           +------------------+
|   Browser   |  ---------------------> |   FastAPI API    |
|   / curl    |                         |   port 8000      |
+-------------+                         +---------+--------+
                                                |
                                                | SQLAlchemy
                                                v
                                      +------------------+
                                      |    PostgreSQL    |
                                      |    port 5432     |
                                      +------------------+
                                                ^
                                                |
                                      +------------------+
                                      |     pgAdmin      |
                                      |     port 5050    |
                                      +------------------+

Docker volumes:
- postgres_data -> PostgreSQL database files
- uploads_data  -> uploaded text files
```

## Folder Structure

```text
docker-document-processor/
├── app/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── crud.py
│   └── analyzer.py
├── uploads/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
├── .gitignore
└── README.md
```

## Technologies Used

- Python 3.11
- FastAPI
- Pydantic
- SQLAlchemy ORM
- PostgreSQL
- pgAdmin
- Docker
- Docker Compose

## Run the Project

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI will be available at:

```text
http://localhost:8000/docs
```

pgAdmin will be available at:

```text
http://localhost:5050
```

Default pgAdmin login:

```text
Email: admin@example.com
Password: admin123
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/upload` | Upload and analyze a `.txt` file |
| `GET` | `/documents` | List all analyzed documents |
| `GET` | `/documents/{id}` | Get one document by ID |
| `DELETE` | `/documents/{id}` | Delete a document record and its uploaded file |

## Upload Example

Create a sample file:

```bash
echo "Docker makes apps portable. Docker Compose runs services together." > sample.txt
```

Upload it:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.txt"
```

Example response:

```json
{
  "message": "Document uploaded and analyzed successfully",
  "id": 1,
  "filename": "sample.txt",
  "word_count": 9,
  "character_count": 66,
  "line_count": 1,
  "top_words": [
    {
      "word": "docker",
      "count": 2
    },
    {
      "word": "makes",
      "count": 1
    }
  ],
  "created_at": "2026-07-19T12:00:00Z"
}
```

## Useful Docker Commands

Start the project:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

Stop containers:

```bash
docker compose down
```

Stop containers and remove volumes:

```bash
docker compose down -v
```

View logs:

```bash
docker compose logs -f api
```

## Screenshots

### Swagger UI

Add a screenshot here after opening `http://localhost:8000/docs`.

```text
screenshots/swagger-ui.png
```

### pgAdmin

Add a screenshot here after opening `http://localhost:5050`.

```text
screenshots/pgadmin.png
```

## Learning Outcomes

- Build a Python API image with a `Dockerfile`
- Run multiple containers with Docker Compose
- Connect FastAPI to PostgreSQL through Docker networking
- Use service names as hostnames inside a Compose network
- Configure services with environment variables
- Persist database data with Docker volumes
- Persist uploaded files with Docker volumes
- Add PostgreSQL healthchecks
- Start the API only after PostgreSQL is healthy
- Create database tables automatically with SQLAlchemy ORM
