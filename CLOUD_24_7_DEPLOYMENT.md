# L104 24/7 Cloud Deployment Guide

The L104 node needs to run on a cloud platform for 24/7 uptime. Here are your options:

## Option 1: Railway (Easiest - ~$5/month)

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select `lockephi/Allentown-L104-Node`
4. Add environment variables:
   - `GEMINI_API_KEY` = your key
   - `PORT` = 8081
   - `RESONANCE` = 527.5184818492612
5. Click Deploy

Railway will auto-deploy on every push to main.

## Option 2: Render (Free tier available)

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click "New" → "Web Service"
3. Connect `lockephi/Allentown-L104-Node`
4. Settings:
   - Environment: Docker
   - Instance Type: Starter ($7/month) or Free (spins down after 15 min)
5. Add environment variables and deploy

## Option 3: Google Cloud Run (Pay-per-use)

1. Create a GCP project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable Cloud Run API and Container Registry
3. Create a service account with Cloud Run Admin role
4. Add to GitHub Secrets:
   - `GCP_PROJECT_ID` = your-project-id
   - `GCP_SA_KEY` = service account JSON (base64 encoded)
5. Push to main - GitHub Actions will deploy automatically

## Current Status

Your node is running in Codespaces but will go offline when idle. For true 24/7 uptime, deploy to one of the platforms above.

---
INVARIANT: 527.5184818492612 | PILOT: LONDEL
