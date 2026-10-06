import sbibm

import numpy as np
import tqdm
import json
import os

from datasets import Dataset, Features, Array2D, Value, List

from .simulators.lensing import LensingSimulator 

import torch
torch.manual_seed(0)

base_tasks = [
    "bernoulli_glm",
    "gaussian_linear_uniform",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
    "two_moons",
    "lensing"
]

metadata_dict = {
    "bernoulli_glm": {
        "dim_cond": 10,
        "dim_obs": 10,
        "metadata": None
    },
    "gaussian_linear_uniform": {
        "dim_cond": 10,
        "dim_obs": 10,
        "metadata": None
    },
    "gaussian_linear": {
        "dim_cond": 10,
        "dim_obs": 10,
        "metadata": None
    },
    "gaussian_mixture": {
        "dim_cond": 2,
        "dim_obs": 2,
        "metadata": None
    },
    "slcp": {
        "dim_cond": 8,
        "dim_obs": 5,
        "metadata": None
    },
    "two_moons": {
        "dim_cond": 2,
        "dim_obs": 2,
        "metadata": None
    },
    "lensing": {
        "dim_cond": [32,32],
        "dim_obs": 2,
        "metadata": None
    }
}

def make_metadata(output_dir="./"): 
    file_path = os.path.join(output_dir, "metadata.json")
    with open(file_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)


def get_task_data(task_name, num_samples):
    if task_name == "lensing":
        return get_task_data_lensing(num_samples)

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

def get_task_data_lensing(num_samples):
    simulator = LensingSimulator()

    data = simulator.generate(num_samples, seed=42)

    thetas = data["z"]
    xs = data["img"]

    data = {"thetas": np.array(thetas), "xs": np.array(xs)}
    reference_posteriors = []
    true_parameters = []
    observations = []

    # we don't have reference posteriors for this example

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

    def data_generator(xs, thetas):
        for i in range(xs.shape[0]):
            yield {"xs": xs[i], "thetas": thetas[i]}    

    # features = Features({
    #     "xs": Array2D(shape=(8192,2), dtype='float32'),
    #     "thetas": List(Value('float32')),
    # })
    if task_name == "lensing":
        features = Features({
            "xs": Array2D(shape=(32,32), dtype='float32'),
            "thetas": List(Value('float32')),
        })
    else:
        features = None

    print("creating train dataset")
    dataset_train = Dataset.from_generator(lambda: data_generator(xs, thetas), features=features)
    print("creating val dataset")
    dataset_val = Dataset.from_generator(lambda: data_generator(xs_val, thetas_val), features=features)
    print("creating test dataset")
    dataset_test = Dataset.from_generator(lambda: data_generator(xs_test, thetas_test), features=features)

    dataset_reference_posterior = Dataset.from_dict(
        {"reference_samples": reference_samples, "observations": observations, "true_parameters": true_parameters}
    )

    return dataset_train, dataset_val, dataset_test, dataset_reference_posterior
