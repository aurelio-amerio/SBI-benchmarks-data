from huggingface_hub import upload_file
from sbi_benchmarks.sbi_tasks import make_dataset

def upload_metadata(file_path: str, repo_name: str):

    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="metadata.json",  # The name of the file in the repo
        repo_id=repo_name,
        repo_type="dataset",
    )

def upload_dataset(repo_name, task_name):
    dataset_train, dataset_val, dataset_test, dataset_reference_posterior = make_dataset(task_name)

    dataset_train.push_to_hub(repo_name, config_name=task_name, split="train", private=False)
    dataset_val.push_to_hub(repo_name, config_name=task_name, split="validation", private=False)
    dataset_test.push_to_hub(repo_name, config_name=task_name, split="test", private=False)
    dataset_reference_posterior.push_to_hub(repo_name, config_name=f"{task_name}_posterior", split="reference_posterior", private=False)