import sbibm

import numpy as np
import tqdm
import json
import os

import torch
torch.manual_seed(0)

from datasets import Dataset

base_tasks = [
    "bernoulli_glm",
    "gaussian_linear_uniform",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
    "two_moons",
]

def make_metadata(output_dir="./"):
    metadata = {}

    pbar = tqdm.tqdm(total=len(base_tasks) )
    for task_name in base_tasks:
        task = sbibm.get_task(task_name)
        task_name = task.name
        dim_cond = task.dim_data
        dim_obs = task.dim_parameters

        metadata[task_name] = {"dim_cond": dim_cond, "dim_obs": dim_obs, "metadata": None}
        pbar.update(1)


    file_path = os.path.join(output_dir, "metadata.json")
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_task_data(task_name, num_samples):
    task = sbibm.get_task(task_name)

    prior = task.get_prior()
    simulator = task.get_simulator()

    thetas = prior(num_samples=num_samples)
    xs = simulator(thetas)

    data = {"thetas": thetas.numpy(), "xs": xs.numpy()}
    reference_posteriors = []
    true_parameters = []
    observations = []
    for i in range(1,11):
        observation = task.get_observation(num_observation=i).numpy()
        reference_posterior = task.get_reference_posterior_samples(num_observation=i).numpy()
        true_params = task.get_true_parameters(num_observation=i).numpy()

        observations.append(observation)
        reference_posteriors.append(reference_posterior)
        true_parameters.append(true_params)

    return data, reference_posteriors, true_parameters, observations


def make_dataset(task_name):

    max_samples = int(1e6)
    num_samples_val =  10_000
    num_samples_test = 10_000

    num_samples = max_samples + num_samples_val + num_samples_test

    data_dict, reference_posteriors, true_parameters, observations = get_task_data(task_name, num_samples)
    
    dtype = np.float32

    xs = data_dict["xs"][: max_samples]
    xs = np.array(xs).astype(dtype)
    thetas = data_dict["thetas"][: max_samples]
    thetas = np.array(thetas).astype(dtype)

    xs_val = data_dict["xs"][max_samples : max_samples + num_samples_val]
    xs_val = np.array(xs_val).astype(dtype)
    thetas_val = data_dict["thetas"][max_samples : max_samples + num_samples_val]
    thetas_val = np.array(thetas_val).astype(dtype)

    xs_test = data_dict["xs"][max_samples + num_samples_val : ]
    xs_test = np.array(xs_test).astype(dtype)
    thetas_test = data_dict["thetas"][max_samples + num_samples_val : ]
    thetas_test = np.array(thetas_test).astype(dtype)

    observations = np.array(observations).astype(dtype)

    reference_samples = np.array(reference_posteriors)
    reference_samples = reference_samples.astype(dtype)

    true_parameters = np.array(true_parameters).astype(dtype)

    dataset_train = Dataset.from_dict({"xs": xs, "thetas": thetas})
    dataset_val = Dataset.from_dict({"xs": xs_val, "thetas": thetas_val})
    dataset_test = Dataset.from_dict({"xs": xs_test, "thetas": thetas_test})
    dataset_reference_posterior = Dataset.from_dict(
        {"reference_samples": reference_samples, "observations": observations, "true_parameters": true_parameters}
    )

    return dataset_train, dataset_val, dataset_test, dataset_reference_posterior
