
import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax

def calculate_ground_state_energy(hamiltonian):
    """
    Calculates exact ground state energy for a PennyLane Hamiltonian.
    
    Args:
        hamiltonian (qml.Hamiltonian): The Hamiltonian to solve.
        
    Returns:
        float: The lowest eigenvalue (ground state energy).
    """
    # Create the matrix representation
    H_mat = qml.matrix(hamiltonian)
    
    # Calculate eigenvalues using JAX's numpy linear algebra
    # eigvalsh is typically faster for Hermitian matrices
    evals = jnp.linalg.eigvalsh(H_mat)
    
    # Return the lowest eigenvalue
    return float(evals[0])

def run_vqe_experiment(circuit_fn, n_qubits, p_layers, n_reps=10, max_steps=1000, learning_rate=0.01, tol=1e-5, optimizer_type='adam'):
    """
    Runs a VQE experiment using JAX for compilation and parallelization.
    
    Args:
        circuit_fn (callable): A function circuit_fn(params, n_qubits) that returns specific expval(H).
        n_qubits (int): Number of qubits.
        p_layers (int): Number of layers in the ansatz.
        n_reps (int): Number of independent training runs to perform in parallel (vmap).
        max_steps (int): Maximum number of optimization steps.
        learning_rate (float or optax.Schedule): Learning rate for the optimizer.
        tol (float): Convergence tolerance for energy differences.
        optimizer_type (str): 'adam' (default) or others if extended.
        
    Returns:
        tuple: (mean_energy, std_energy, mean_iterations)
    """
    
    cols = 0
    fn_name = getattr(circuit_fn, '__name__', '').lower()
    
    if 'equivariant' in fn_name and 'non' not in fn_name:
        # Standard equivariant usually (p, 2) for both models (beta, gamma)
        cols = 2
    elif 'nonequivariant' in fn_name or 'non_equivariant' in fn_name:
        pass

    return _run_vqe_internal(circuit_fn, n_qubits, p_layers, n_reps, max_steps, learning_rate, tol)

def _run_vqe_internal(circuit_fn, n_qubits, p_layers, n_reps, max_steps, learning_rate, tol):
    pass

def run_vqe_experiment(circuit_fn, n_qubits, p_layers, param_cols, n_reps=10, max_steps=1000, learning_rate=0.01, tol=1e-5):
    """
    Runs a VQE experiment using JAX for compilation and parallelization.
    
    Args:
        circuit_fn (callable): A function circuit_fn(params, n_qubits) that returns specific expval(H).
        n_qubits (int): Number of qubits.
        p_layers (int): Number of layers in the ansatz.
        param_cols (int): Number of parameter columns per layer (e.g., 2 for beta/gamma).
        n_reps (int): Number of independent training runs to perform in parallel (vmap).
        max_steps (int): Maximum number of optimization steps.
        learning_rate (float or optax.Schedule): Learning rate for the optimizer.
        tol (float): Convergence tolerance for energy differences.
        
    Returns:
        tuple: (mean_energy, std_energy, mean_iterations)
    """
    param_shape = (n_reps, p_layers, param_cols)
    
    # Setup Optimizer
    if isinstance(learning_rate, (float, int)):
        optimizer = optax.adam(learning_rate=learning_rate)
    else:
        # Assume it's a schedule or valid optax object
        optimizer = optax.adam(learning_rate=learning_rate)
        
    # Define Single Step (Unbatched)
    def single_step_fn(carry, t):
        # t is iteration index, sometimes needed for schedules
        params, opt_state = carry

        # Define loss with fixed n_qubits for this specific call  
        def loss_fn(p):
            return circuit_fn(p, n_qubits)
            
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    # Vectorize over repetitions (n_reps)
    # in_axes=(0, 0) means we expect batched params and batched opt_state
    vmapped_step = jax.vmap(single_step_fn, in_axes=((0, 0), None))

 # Compile the entire training loop (Scan)
    @jax.jit
    def optimize_circuit(init_params, init_opt_state):
        """Instead of a Python for loop communicating with the GPU 1000 times (1000 round-trips), 
        the entire loop is compiled into a single XLA kernel. The Python interpreter only fires once.   """
        # scan runs `vmapped_step` `max_steps` times
        steps = jnp.arange(max_steps)
        final_carry, history = jax.lax.scan(vmapped_step, (init_params, init_opt_state), steps)
        return history # Shape: (max_steps, n_reps)

    # Initialization
    # Use a seed that depends on args to be deterministic but varied
    print(f"Compiling & Running {circuit_fn.__name__} (N={n_qubits}, p={p_layers})...")
    key = jax.random.PRNGKey(42 + p_layers * 100 + n_qubits)
    init_params = jax.random.normal(key, param_shape) * 0.1
    
    # Initialize optimizer state (batched)
    init_opt_state = jax.vmap(optimizer.init)(init_params)
    
    # Execute Training
    loss_history = optimize_circuit(init_params, init_opt_state)
    
    # Post-Process results on CPU (Convergence Check)
    loss_history_np = np.array(loss_history)
    final_energies = []
    iterations_list = []
    
    for rep in range(n_reps):
        trace = loss_history_np[:, rep]
        final_energies.append(trace[-1])
        
        # Calculate convergence
        # First step where |loss[t] - loss[t-1]| < tol
        diffs = np.abs(np.diff(trace))

        if len(diffs) > 10:
            smoothed_diffs = np.convolve(diffs, np.ones(10)/10, mode='valid')
            converged_indices = np.where(smoothed_diffs < tol)[0]
        else:
            converged_indices = np.where(diffs < tol)[0]
            
        if len(converged_indices) > 0:
            # +10 or +1 because of diff/convolve shift
            iterations_list.append(converged_indices[0] + 1) 
        else:
            iterations_list.append(max_steps)
            
    return np.mean(final_energies), np.std(final_energies), np.mean(iterations_list)
import matplotlib.pyplot as plt
import numpy as np

def plot_results(p_values, n_experiment, gs_energy, eq_data, neq_data, model_name):
    """
    Plots the VQE convergence results: Energy Error and Iterations to Convergence.
    
    Args:
        p_values (list): List of p (layer) values.
        n_experiment (int): Number of qubits/system size.
        gs_energy (float): The exact ground state energy.
        eq_data (dict): Dictionary containing "energy", "std", "iters" for Equivariant ansatz.
        neq_data (dict): Dictionary containing "energy", "std", "iters" for Non-Equivariant ansatz.
        model_name (str): Name of the model (e.g., "TFIM", "Heisenberg Model").
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # Top Panel: Mean Energy Error with Standard Deviation
    # Equivariant
    ax1.errorbar(p_values, np.array(eq_data["energy"]) - gs_energy, 
                 yerr=eq_data["std"], fmt='o-', label="Equivariant", color='tab:blue', capsize=5)
    
    # Non-Equivariant
    ax1.errorbar(p_values, np.array(neq_data["energy"]) - gs_energy, 
                 yerr=neq_data["std"], fmt='s-', label="Non-Equivariant", color='tab:orange', capsize=5)
    
    ax1.set_ylabel("Energy Error ($E - E_{GS}$)")
    ax1.set_title(f"{model_name} (N={n_experiment})")
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Log scale helps see the small error of the equivariant case at high depth
    # ax1.set_yscale('log') 
    
    # Bottom Panel: Iterations to Convergence
    ax2.plot(p_values, eq_data["iters"], 'o-', label="Equivariant", color='tab:blue')
    ax2.plot(p_values, neq_data["iters"], 's-', label="Non-Equivariant", color='tab:orange')
    ax2.set_ylabel("Iterations to Convergence")
    ax2.set_xlabel("Layers ($p$)")
    ax2.set_title(f"Convergence Speed")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
