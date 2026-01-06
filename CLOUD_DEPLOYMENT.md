# Cloud Deployment Guide for L104 Sovereign Node

## Overview
This document outlines the cloud deployment strategy for the L104 Sovereign Node application.

## Current State
The L104 Sovereign Node is a FastAPI-based application that includes:
- Gemini API relay with health/metrics
- Rate limiting capabilities
- Memory store (SQLite)
- Multiple L104 modules for AI/ML processing
- Keep-alive mechanisms

## Cloud Deployment Considerations

### 1. Platform Options
- **Google Cloud Platform (GCP)**: Natural fit given Gemini integration
  - Cloud Run for serverless deployment
  - Cloud SQL for persistent storage
  - Cloud Scheduler for keep-alive jobs
  
- **AWS**: Alternative deployment
  - ECS/Fargate for containerized deployment
  - RDS for database
  - EventBridge for scheduled tasks
  
- **Azure**: Another alternative
  - Azure Container Instances
  - Azure Database for PostgreSQL
  - Azure Functions for scheduled tasks

### 2. Required Environment Variables
```bash
GEMINI_API_KEY=<your-gemini-key>
GITHUB_TOKEN=<your-github-token>
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_ENDPOINT=:streamGenerateContent
ENABLE_FAKE_GEMINI=0
MEMORY_DB_PATH=/data/memory.db
RAMNODE_DB_PATH=/data/ramnode.db
SELF_BASE_URL=<production-url>
```

### 3. Containerization
Create a `Dockerfile` for containerized deployment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8081
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
```

### 4. Persistent Storage
- SQLite databases need to be mounted on persistent volumes
- Consider migrating to cloud-native database solutions for production
- Backup strategy for `memory.db` and `ramnode.db`

### 5. Scaling Considerations
- Horizontal scaling may require shared state management
- Rate limiting needs to be coordinated across instances
- Consider Redis for distributed rate limiting

### 6. Monitoring & Observability
- Health endpoint: `/health`
- Metrics endpoint: `/metrics`
- Set up cloud monitoring/alerting
- Log aggregation service

### 7. Security
- API keys stored in secret management service
- HTTPS/TLS termination at load balancer
- Network security groups/firewall rules
- Regular security updates

### 8. CI/CD Pipeline
- Automated testing on PR
- Container image build on merge
- Deployment to staging environment
- Production deployment with approval

## Next Steps
1. Choose cloud platform based on requirements
2. Set up cloud infrastructure (IaC with Terraform/CloudFormation)
3. Configure CI/CD pipeline
4. Implement monitoring and alerting
5. Test deployment in staging environment
6. Deploy to production

## Notes
- The keep-alive workflow (`keep_alive.yml`) should be adapted to work with cloud platform's native scheduling
- Consider serverless options to reduce costs during low traffic periods
- Ensure all L104 modules are compatible with containerized environment
