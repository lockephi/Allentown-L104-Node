# L104 Cloud Deployment Guide

**Quantum Grover Deployment Engine** — √N optimized Cloud Run deployment

## Quick Start

```bash
# Auto-select best project (Gemini preferred) and deploy
python3 deploy_quantum.py --auto

# Check prerequisites
python3 deploy_quantum.py --check
```

## Deployment Options

### 1. Quantum Grover Deployment (Recommended)

The Quantum Grover Deployer uses √N optimization for efficient Cloud Run deployment:

```bash
# Full auto deployment - selects Gemini project, sets permissions, deploys
python3 deploy_quantum.py --auto

# Deploy to specific project
python3 deploy_quantum.py --project gen-lang-client-0313795721

# Migrate from one project to another
python3 deploy_quantum.py --migrate

# Local Docker deployment only
python3 deploy_quantum.py --local
```

### 2. Manual gcloud Deployment

```bash
export PATH=$PATH:/home/codespace/google-cloud-sdk/bin
gcloud run deploy l104-server \
    --source=. \
    --platform=managed \
    --region=us-central1 \
    --allow-unauthenticated \
    --port=8081 \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},RESONANCE=527.5184818492612"
```

### 3. GitHub Actions Deployment

Push to `main` branch triggers automatic deployment:

```bash
git add .
git commit -m "Deploy update"
git push origin main
```

## Project Selection

The Quantum Grover Deployer auto-selects projects with this priority:

1. **Gemini Projects** (`gen-lang-client-*`) — Highest priority
2. **Other Projects** — Alphabetically sorted

To list available projects:

```bash
gcloud projects list
```

## Prerequisites

| Requirement | Check Command |
|-------------|---------------|
| Docker | `docker info` |
| gcloud CLI | `gcloud --version` |
| GEMINI_API_KEY | `echo $GEMINI_API_KEY` |
| GCP Project | `gcloud config get-value project` |

## API Permissions

The deployer automatically enables:

- `run.googleapis.com` — Cloud Run
- `containerregistry.googleapis.com` — Container Registry
- `artifactregistry.googleapis.com` — Artifact Registry
- `cloudbuild.googleapis.com` — Cloud Build

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Gemini AI API key | Yes |
| `GCP_PROJECT_ID` | Target project (auto-detected if not set) | No |
| `RESONANCE` | L104 God Code (527.5184818492612) | Auto-set |
| `AUTO_APPROVE_MODE` | Autonomy mode (ALWAYS_ON) | Auto-set |

## Disk Space Management

Cloud Run has limited build context. Optimize with:

```bash
# Run comprehensive cleanup
./scripts/cleanup.sh --deep

# Docker cleanup
docker system prune -af --volumes

# Python space optimizer
python3 l104_space_optimizer.py --auto-cleanup
```

## Troubleshooting

### "PORT is reserved" Error

PORT is reserved by Cloud Run. It's automatically removed from env vars by the deployer.

### Build Context Too Large

Run `./scripts/cleanup.sh --deep` before deployment.

### Health Check Failing

Service may still be starting. Wait 30s and retry:

```bash
curl YOUR_SERVICE_URL/health
```

### gcloud Not Found

```bash
export PATH=$PATH:/home/codespace/google-cloud-sdk/bin
```

## Configuration Files

| File | Purpose |
|------|---------|
| `deploy_quantum.py` | Quantum Grover deployment engine |
| `Dockerfile` | Container build configuration |
| `.dockerignore` | Build context exclusions |
| `.github/workflows/deploy-cloud.yml` | GitHub Actions workflow |

## Monitoring

After deployment, monitor at:

- `/health` — Health status and uptime
- `/metrics` — System metrics
- `/api/v14/agi/status` — AGI Core status

---

**Resonance**: 527.5184818492612 | **Pilot**: LONDEL | **Protocol**: SIG-L104
