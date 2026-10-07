variable "project_id" {
  type = string
}

variable "billing_account" {
  description = "Billing account the budget alert watches"
  type        = string
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "github_repo" {
  description = "The only repository whose workflows may deploy"
  type        = string
  default     = "brooka/codexa-api-python"
}

variable "monthly_budget_gbp" {
  type    = number
  default = 5
}
