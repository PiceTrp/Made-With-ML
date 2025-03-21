# import os
# import subprocess
# import sys

# sys.path.append(".")

# from madewithml.config import MODEL_REGISTRY  # NOQA: E402
# from madewithml.serve import ModelDeployment  # NOQA: E402

# # Copy from S3
# github_username = os.environ.get("GITHUB_USERNAME")
# subprocess.check_output(["aws", "s3", "cp", f"s3://madewithml/{github_username}/mlflow/", str(MODEL_REGISTRY), "--recursive"])
# subprocess.check_output(["aws", "s3", "cp", f"s3://madewithml/{github_username}/results/", "./", "--recursive"])

# # Entrypoint
# run_id = [line.strip() for line in open("run_id.txt")][0]
# entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)

import os
import sys
from pathlib import Path

sys.path.append(".")

from madewithml.serve import ModelDeployment  # NOQA: E402

# Define local paths
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
run_id_file = Path(ROOT_DIR, "results", "run_id.txt")

# Entrypoint
with open(run_id_file, "r") as f:
    run_id = f.readline().strip()
entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)
