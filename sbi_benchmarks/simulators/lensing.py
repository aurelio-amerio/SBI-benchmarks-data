import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm # Requires: pip install tqdm
import functools

class LensingSimulator:
    """
    Usage 
    ------ 
    sim = LensingSimulator(resolution=32)

    # Generate 5,000 simulations
    data = sim.generate(n_sims=5000, batch_size=500)

    print("Output Shapes:")
    print("z:  ", data['z'].shape)   # (5000, 2)
    print("mu: ", data['mu'].shape)  # (5000, 32, 32)
    print("img:", data['img'].shape) # (5000, 32, 32)

    i=100
    plt.imshow(data['img'][i], cmap='gray')
    plt.title(f"z = {data['z'][i]}")
    plt.show()
"""
    def __init__(self, resolution=32):
        # Setup grid (static data)
        x = jnp.linspace(-2, 2, resolution)
        self.X, self.Y = jnp.meshgrid(x, x)

    # --- Physics Methods (Same as before) ---
    def _get_z(self, key):
        minval = jnp.array([0.1, 0.01])
        scale = jnp.array([1.0, 0.3])
        return minval + random.uniform(key, (2,)) * scale

    def _get_mu(self, key, z):
        r, w = z
        k_pos, k_lines = random.split(key)
        pos = random.uniform(k_pos, (2,), minval=-1, maxval=1)
        x0, y0 = pos[0], pos[1]
        
        R = jnp.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        mu = jnp.exp(-(R - r)**2 / w**2 / 2)

        xr_batch = random.uniform(k_lines, (20, 2))

        def add_line(xr):
            term = (self.X * xr[0] + self.Y * (1 - xr[0]) - xr[1])
            return 0.8 * jnp.exp(-term**2 / 0.01**2)

        distortions = jax.vmap(add_line)(xr_batch)
        mu = mu + jnp.sum(distortions, axis=0)
        mu = mu - jnp.mean(mu)
        mu = mu / jnp.std(mu)
        return mu

    def _get_img(self, key, mu):
        return mu + random.normal(key, mu.shape) * 0.3

    # --- Sampling Logic ---
    def _sample_one(self, key):
        """Generates a single simulation."""
        k_z, k_mu, k_img = random.split(key, 3)
        z = self._get_z(k_z)
        mu = self._get_mu(k_mu, z)
        img = self._get_img(k_img, mu)
        return {'z': z, 'mu': mu, 'img': img}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_batch(self, keys):
        """
        Generates a batch of simulations in parallel using vmap.
        JIT compiled for speed.
        """
        return jax.vmap(self._sample_one)(keys)

    def generate(self, n_sims, batch_size=1000, seed=42):
        """
        Main method to generate N simulations with a progress bar.
        """
        master_key = random.PRNGKey(seed)
        # Prepare all keys upfront
        all_keys = random.split(master_key, n_sims)
        
        results = []
        
        # Calculate number of batches
        num_batches = int(jnp.ceil(n_sims / batch_size))
        
        # Loop with progress bar
        for i in tqdm(range(num_batches), desc="Simulating"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_sims)
            
            # Slice keys for this batch
            batch_keys = all_keys[start_idx:end_idx]
            
            # Run simulation for this batch
            batch_res = self._sample_batch(batch_keys)
            batch_res['img'].block_until_ready()
            
            # Move results to CPU (optional, but saves GPU memory for next batch)
            # batch_res = jax.device_get(batch_res) 
            results.append(batch_res)

        # Concatenate all batches into final arrays
        # jax.tree_map applies concatenation to every key (z, mu, img) in the dict
        final_data = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)
        
        return final_data

# # --- Usage ---
# sim = JaxSimulator()

# # Generate 5,000 simulations
# data = sim.generate(n_sims=5000, batch_size=500)

# print("Output Shapes:")
# print("z:  ", data['z'].shape)   # (5000, 2)
# print("mu: ", data['mu'].shape)  # (5000, 32, 32)
# print("img:", data['img'].shape) # (5000, 32, 32)