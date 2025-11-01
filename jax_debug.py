# In check_jax_internals.py
import os
import jax
import jaxlib
import sys
import subprocess

print(f"--- JAX Internal Diagnostics ---")
print(f"Python Executable: {sys.executable}")
print(f"JAX version: {jax.__version__}")
print(f"JAXLIB version: {jaxlib.__version__}")

print("\n--- Environment Variables (from Python) ---")
# Check for interfering environment variables
ld_path = os.environ.get('LD_LIBRARY_PATH', 'NOT SET')
print(f"LD_LIBRARY_PATH: {ld_path}")
path = os.environ.get('PATH', 'NOT SET')
print(f"PATH: {path}")

# Check if an old ptxas is on the path
print("\n--- Compiler Check (which ptxas) ---")
try:
    # This will find the first 'ptxas' on the system PATH
    ptxas_path = subprocess.check_output(['which', 'ptxas']).decode('utf-8').strip()
    print(f"System 'ptxas' found at: {ptxas_path}")
    if 'cuda-11' in ptxas_path:
        print("WARNING: 'ptxas' from an old CUDA 11 toolkit is on your PATH. This is likely the problem.")
except Exception:
    print("System 'ptxas' not found on PATH. (This is often OK for modern JAX).")

print("\n--- JAX Device and PRNG Test ---")
try:
    print(f"JAX Devices: {jax.devices()}")
    print("Attempting JAX PRNGKey (this is the operation that fails in your script)...")
    key = jax.random.PRNGKey(0)
    _ = key.block_until_ready()
    print("JAX PRNGKey test: SUCCESS")
    print(f"Default Backend: {jax.default_backend()}")
except Exception as e:
    print("JAX PRNGKey test: FAILED")
    print(e)
print(f"---------------------------------")
