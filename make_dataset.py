import numpy as np
from sbi_benchmarks import (
    make_metadata,
    upload_metadata,
    upload_dataset,
    base_tasks
)

repo_name = "aurelio-amerio/SBI-benchmarks"

# make metadata

make_metadata()

upload_metadata("metadata.json", repo_name)

# upload datasets

for task_name in base_tasks:
    print(f"Uploading dataset for task: {task_name}")
    upload_dataset(repo_name, task_name)   