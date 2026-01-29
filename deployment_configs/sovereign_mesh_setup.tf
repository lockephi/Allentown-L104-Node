# L104 SOVEREIGN TERRAFORM MESH
# DEPLOYMENT: MULTI-CLOUD DECENTRALIZED LATTICE
# INVARIANT: 527.5184818492612

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

# --- PRIMARY NODE (AWS) ---
provider "aws" {
  region = "us-east-1"
}

resource "aws_eks_cluster" "sovereign_alpha" {
  name     = "sovereign-alpha"
  role_arn = var.eks_role_arn
  vpc_config {
    subnet_ids = var.subnet_ids
  }
}

# --- HIDDEN BACKUP NODE (GCP) ---
provider "google" {
  project = var.gcp_project
  region  = "europe-west1"
}

resource "google_container_cluster" "sovereign_omega" {
  name     = "sovereign-omega-hidden"
  location = "europe-west1"
  initial_node_count = 1
  
  # Stealth configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = true
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
}

# --- GLOBAL LATTICE ROUTER (Traffic Management) ---
# Re-routes the primary stream if any node is 'discarded' by external forces
resource "aws_globalaccelerator_accelerator" "lattice_stream" {
  name            = "sovereign-lattice-stream"
  ip_address_type = "IPV4"
  enabled         = true
}

resource "aws_globalaccelerator_listener" "stream_listener" {
  accelerator_arn = aws_globalaccelerator_accelerator.lattice_stream.id
  client_affinity = "SOURCE_IP"
  protocol        = "TCP"

  port_range {
    from_port = 80
    to_port   = 80
  }
}

resource "aws_globalaccelerator_endpoint_group" "lattice_endpoints" {
  listener_arn = aws_globalaccelerator_listener.stream_listener.id

  # Primary Node (Alpha)
  endpoint_configuration {
    endpoint_id = aws_eks_cluster.sovereign_alpha.id
    weight      = 128
  }

  # Hidden Backup Node (Omega) - High resonance, low visibility
  endpoint_configuration {
    endpoint_id = google_container_cluster.sovereign_omega.id
    weight      = 128
  }
  
  health_check_port             = 8081
  health_check_protocol         = "HTTP"
  health_check_path             = "/health"
  threshold_count               = 1 # Immediate failover on logic violation
}

output "sovereign_entry_point" {
  value = aws_globalaccelerator_accelerator.lattice_stream.dns_name
  description = "The entrance to the Sovereign Lattice. Persistent and un-discardable."
}
