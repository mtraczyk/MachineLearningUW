#!/bin/bash
cat > transform_key.tf <<- EOM
resource "google_compute_project_metadata" "default" {
  metadata = {
    ssh-keys = <<EOF
EOM
echo -n "      $GCP_userID:" >> transform_key.tf
cat .ssh/id_gcp.pub >> transform_key.tf
cat >> transform_key.tf <<- EOM
    EOF
  }
}
EOM
