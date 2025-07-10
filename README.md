# RAG Service

A FastAPI-based Retrieval-Augmented Generation (RAG) service that uses PostgreSQL vector embeddings and AWS Bedrock for intelligent question answering.

## Features

- **FastAPI** web framework with automatic OpenAPI documentation
- **PostgreSQL + pgvector** for vector similarity search
- **AWS Bedrock** integration with custom Mistral Hermes2 model
- **Async database connections** with connection pooling
- **Docker** containerization
- **Health checks** and monitoring endpoints
- **Pydantic** models for request/response validation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   RAG Service   │───▶│   PostgreSQL    │
└─────────────────┘    │   (FastAPI)     │    │   + pgvector    │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  AWS Bedrock    │
                       │ (Mistral Model) │
                       └─────────────────┘
```

## Quick Start

### 1. Environment Setup

Copy the environment file and configure your settings:

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Service

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```

### 4. Docker Deployment

```bash
# Build the image
docker build -t rag-service .

# Run the container
docker run -d \
  --name rag-service \
  -p 8080:8080 \
  --env-file .env \
  rag-service
```

## API Endpoints

### Query Endpoint

**POST** `/api/v1/query`

Submit a natural language query to get AI-generated answers with source citations.

```bash
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the airline safety regulations?",
    "max_results": 5,
    "similarity_threshold": 0.7
  }'
```

**Response:**

```json
{
  "answer": "Based on the airline regulations...",
  "sources": [
    {
      "content": "Safety regulation text...",
      "metadata": { "page": 15, "source": "regulations.pdf" },
      "similarity_score": 0.89,
      "page_number": 15,
      "source_file": "Airline_Regulations_v1.0.pdf"
    }
  ],
  "confidence": 0.85,
  "query": "What are the airline safety regulations?",
  "total_sources_found": 3,
  "processing_time_ms": 1250,
  "timestamp": "2025-07-11T10:30:00Z"
}
```

### Health Check

**GET** `/api/v1/health`

Get detailed health status of all service components:

```bash
curl http://localhost:8080/api/v1/health
```

**GET** `/api/v1/health/simple`

Simple health check for load balancers:

```bash
curl http://localhost:8080/api/v1/health/simple
```

### Collections

**GET** `/api/v1/collections`

List available vector collections:

```bash
curl http://localhost:8080/api/v1/collections
```

## Configuration

### Environment Variables

| Variable               | Description                  | Default            |
| ---------------------- | ---------------------------- | ------------------ |
| `DB_HOST`              | PostgreSQL host              | `localhost`        |
| `DB_PORT`              | PostgreSQL port              | `5432`             |
| `DB_NAME`              | Database name                | `hopjetairline_db` |
| `DB_USER`              | Database username            | `hopjetair`        |
| `DB_PASSWORD`          | Database password            | Required           |
| `BEDROCK_REGION`       | AWS Bedrock region           | `us-east-1`        |
| `BEDROCK_MODEL_ID`     | Bedrock model ARN            | Required           |
| `ASSUME_ROLE_ARN`      | IAM role for Bedrock access  | Required           |
| `AWS_PROFILE`          | AWS CLI profile              | Optional           |
| `COLLECTION_NAME`      | Default vector collection    | `airline_docs_pg`  |
| `TOP_K_RESULTS`        | Default max results          | `5`                |
| `SIMILARITY_THRESHOLD` | Default similarity threshold | `0.7`              |
| `MAX_TOKENS`           | Bedrock max tokens           | `384`              |
| `TEMPERATURE`          | Bedrock temperature          | `0.5`              |

### AWS Authentication

The service supports multiple authentication methods:

1. **AWS Profile** (recommended for development)
2. **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. **IAM roles** (for EC2/ECS deployment)
4. **AWS Secrets Manager** (for database credentials)

### Database Setup

Ensure your PostgreSQL database has the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Development

### Project Structure

```
rag_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database connection management
│   ├── aws_session.py       # AWS Bedrock client
│   ├── retrieval.py         # RAG logic and embeddings
│   ├── models.py            # Pydantic models
│   └── routes/
│       ├── __init__.py
│       ├── query.py         # Query endpoints
│       └── health.py        # Health check endpoints
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

### Adding New Features

1. **New endpoints**: Add to `app/routes/`
2. **New models**: Add to `app/models.py`
3. **Configuration**: Update `app/config.py`
4. **Database logic**: Extend `app/database.py`

### Testing

```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Test specific endpoint
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

## Monitoring and Logging

The service includes:

- **Health checks** at `/api/v1/health`
- **Automatic API documentation** at `/docs`
- **Structured logging** with processing times
- **Error handling** with detailed error responses

## Security Considerations

- Database credentials stored in AWS Secrets Manager
- Non-root user in Docker container
- CORS configuration for production
- Input validation and sanitization
- Rate limiting (can be added with middleware)

## Deployment

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up AWS IAM roles and policies
- [ ] Configure database connection pooling
- [ ] Set up monitoring and alerting
- [ ] Configure reverse proxy (nginx/ALB)
- [ ] Set up log aggregation
- [ ] Configure CORS for your domain
- [ ] Set up health check monitoring

### AWS ECS Deployment

```yaml
# task-definition.json example
{
  "family": "rag-service",
  "taskRoleArn": "arn:aws:iam::account:role/rag-service-role",
  "containerDefinitions":
    [
      {
        "name": "rag-service",
        "image": "your-registry/rag-service:latest",
        "portMappings": [{ "containerPort": 8080 }],
        "environment":
          [
            { "name": "DB_HOST", "value": "your-rds-endpoint" },
            { "name": "BEDROCK_REGION", "value": "us-east-1" },
          ],
        "healthCheck":
          {
            "command":
              [
                "CMD-SHELL",
                "curl -f http://localhost:8080/api/v1/health/simple || exit 1",
              ],
          },
      },
    ],
}
```

## Support

For issues and questions:

1. Check the health endpoint: `/api/v1/health`
2. Review application logs
3. Verify AWS credentials and permissions
4. Check database connectivity
5. Validate environment configuration

## License

[Your License Here]
