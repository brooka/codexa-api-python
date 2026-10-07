# Codexa API on Cloud Run: scales to zero, so it costs nothing while idle.
# Terraform creates the infrastructure once; GitHub Actions builds and deploys each new image.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each = toset([
    "apikeys.googleapis.com",
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "generativelanguage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── Storage ──────────────────────────────────────────────────────────────

# Container images. CI pushes one per deploy; keep the two newest so storage stays small.
resource "google_artifact_registry_repository" "images" {
  repository_id = "codexa"
  location      = var.region
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-two-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 2
    }
  }
  cleanup_policies {
    id     = "delete-the-rest"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }

  depends_on = [google_project_service.apis]
}

# The packs and the embedding model (0.8 GB, too big for git). CI copies them into the image.
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-data"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  depends_on                  = [google_project_service.apis]
}

# Secret values are added with gcloud, so they never appear in Terraform state.
resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(["gemini-api-key", "ask-api-key"])
  secret_id = each.value
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# ── The service ──────────────────────────────────────────────────────────

# The service's identity: it may read its two secrets and nothing else.
resource "google_service_account" "runtime" {
  account_id   = "codexa-api"
  display_name = "Codexa API (Cloud Run)"
}

resource "google_secret_manager_secret_iam_member" "runtime" {
  for_each  = google_secret_manager_secret.secrets
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name                = "codexa-api"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = 0 # nothing runs, and nothing is billed, between requests
      max_instance_count = 1 # caps the cost of a traffic spike
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello" # placeholder: CI deploys the real image

      resources {
        limits = {
          cpu    = "1"
          memory = "4Gi"
        }
        cpu_idle          = true # CPU only while handling a request
        startup_cpu_boost = true # faster cold starts
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["gemini-api-key"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "CODEXA_ASK_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["ask-api-key"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  # CI owns the image from the first deploy on.
  lifecycle {
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime]
}

# Anyone may call it: /search is free to run, and /ask needs the API key.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Deploys from GitHub Actions, without stored keys ─────────────────────

resource "google_service_account" "deployer" {
  account_id   = "github-deployer"
  display_name = "GitHub Actions deployer"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repo}'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Only workflows in the one repository may act as the deployer.
resource "google_service_account_iam_member" "deployer_from_github" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# What a deploy needs: read the data, push the image, roll out a revision that runs as the service's identity.
resource "google_storage_bucket_iam_member" "deployer_reads_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_artifact_registry_repository_iam_member" "deployer_pushes_images" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_deploys" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_runs_as_service" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

# ── Cost alert ───────────────────────────────────────────────────────────

# Emails the billing account's admins at 50%, 90% and 100% of the monthly budget.
resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "${var.project_id} monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = "GBP"
      units         = tostring(var.monthly_budget_gbp)
    }
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  depends_on = [google_project_service.apis]
}
