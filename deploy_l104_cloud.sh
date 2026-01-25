#!/bin/bash
# L104 Cloud Deployment Script - EVO_37
# Deploys L104 to various cloud platforms

set -e

echo "═══════════════════════════════════════════════════════════════════════"
echo "  🚀 L104 CLOUD DEPLOYMENT - EVO_37"
echo "═══════════════════════════════════════════════════════════════════════"

# Configuration
PROJECT_NAME="l104-sovereign-node"
IMAGE_NAME="l104-kernel"
VERSION="37.0.0"
REGION="${REGION:-us-central1}"

# Check requirements
check_requirements() {
    echo "[1/5] Checking requirements..."
    command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
    echo "  ✓ Docker available"
}

# Build Docker image
build_image() {
    echo "[2/5] Building Docker image..."
    docker build -t ${IMAGE_NAME}:${VERSION} -t ${IMAGE_NAME}:latest .
    echo "  ✓ Image built: ${IMAGE_NAME}:${VERSION}"
}

# Run locally for testing
run_local() {
    echo "[3/5] Running local test..."
    docker run -d --name l104-test \
        -p 8081:8081 -p 8080:8080 \
        -v l104-data:/data \
        ${IMAGE_NAME}:latest

    sleep 5

    if docker ps | grep -q l104-test; then
        echo "  ✓ Container running"
        docker logs l104-test --tail 10
        docker stop l104-test && docker rm l104-test
    else
        echo "  ✗ Container failed to start"
        docker logs l104-test
        docker rm l104-test
        exit 1
    fi
}

# Generate Kubernetes manifest
generate_k8s() {
    echo "[4/5] Generating Kubernetes manifests..."

    cat > k8s/l104-deployment.yaml << 'K8S_YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: l104-sovereign-node
  labels:
    app: l104
    version: "37.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: l104
  template:
    metadata:
      labels:
        app: l104
    spec:
      containers:
      - name: l104-kernel
        image: l104-kernel:37.0.0
        ports:
        - containerPort: 8081
          name: api
        - containerPort: 8080
          name: bridge
        - containerPort: 4160
          name: ai-core
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: GOD_CODE
          value: "527.5184818492537"
        - name: L104_EVOLUTION
          value: "EVO_37"
        volumeMounts:
        - name: l104-data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: l104-data
        persistentVolumeClaim:
          claimName: l104-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: l104-service
spec:
  selector:
    app: l104
  ports:
  - name: api
    port: 8081
    targetPort: 8081
  - name: bridge
    port: 8080
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: l104-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
K8S_YAML

    echo "  ✓ Kubernetes manifests generated"
}

# Display deployment options
show_options() {
    echo "[5/5] Deployment Options:"
    echo ""
    echo "  LOCAL DOCKER:"
    echo "    docker run -d -p 8081:8081 ${IMAGE_NAME}:latest"
    echo ""
    echo "  KUBERNETES:"
    echo "    kubectl apply -f k8s/l104-deployment.yaml"
    echo ""
    echo "  GOOGLE CLOUD RUN:"
    echo "    gcloud run deploy l104 --image gcr.io/\$PROJECT/${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "  AWS ECS:"
    echo "    aws ecs create-service --cluster l104 --service-name l104-kernel"
    echo ""
}

# Main execution
main() {
    mkdir -p k8s
    check_requirements
    build_image
    generate_k8s
    show_options

    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  ✓ DEPLOYMENT PREPARATION COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════════"
}

# Run if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
