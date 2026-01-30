#!/usr/bin/env python3
"""
Fast Experimental Verification of Theoretical Results
Optimized version with vectorized operations

Author: Panda 🐼
Date: 2025-01-29
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

np.random.seed(42)
os.makedirs('/home/swei20/ips_unlabeled_learning/experiments/results', exist_ok=True)
os.makedirs('/home/swei20/ips_unlabeled_learning/experiments/figures', exist_ok=True)

print("="*60)
print("Fast Theoretical Verification Experiments 🐼")
print("="*60)

#%% Vectorized IPS Simulator
def generate_ensemble_fast(N, M, L, T_total, sigma=0.1, dt=0.01):
    """Generate ensemble data using vectorized simulation"""
    times = np.linspace(0, T_total, L)
    ensembles = []
    
    for l in range(L):
        if l == 0:
            X = np.random.randn(M, N)  # (M, N) for 1D
        else:
            steps = max(1, int((times[l] - times[l-1]) / dt))
            X = ensembles[-1].copy()
            for _ in range(steps):
                # Vectorized drift computation
                drift = -X  # -∇V = -x for V = 0.5*x²
                
                # Interaction: -1/N * Σ_j ∇Φ(x_i - x_j)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            r = X[:, i] - X[:, j]  # (M,)
                            drift[:, i] -= (-np.exp(-np.abs(r)) * np.sign(r)) / N
                
                dW = np.random.randn(M, N) * np.sqrt(dt)
                X = X + drift * dt + sigma * dW
        
        ensembles.append(X)
    
    return ensembles, times


def compute_loss_fast(a, ensembles, times, sigma=0.1):
    """Vectorized loss computation"""
    L = len(times)
    M, N = ensembles[0].shape
    total_loss = 0.0
    
    for l in range(L - 1):
        X_t = ensembles[l]  # (M, N)
        X_t1 = ensembles[l + 1]
        dt = times[l + 1] - times[l]
        
        # Dissipation: E[|drift|²]
        dissip = 0.0
        for i in range(N):
            drift_i = X_t[:, i].copy()  # ∇V = x
            for j in range(N):
                r = X_t[:, i] - X_t[:, j]
                drift_i += a * (-np.exp(-np.abs(r)) * np.sign(r)) / N
            dissip += np.mean(drift_i**2)
        dissip = dissip / N * dt
        
        # Energy at t and t+1
        E_t = 0.5 * np.mean(np.sum(X_t**2, axis=1)) / N
        E_t1 = 0.5 * np.mean(np.sum(X_t1**2, axis=1)) / N
        
        # Interaction energy
        for i in range(N):
            for j in range(N):
                E_t += a * np.mean(np.exp(-np.abs(X_t[:, i] - X_t[:, j]))) / (2*N*N)
                E_t1 += a * np.mean(np.exp(-np.abs(X_t1[:, i] - X_t1[:, j]))) / (2*N*N)
        
        total_loss += dissip - 2 * (E_t1 - E_t)
    
    return total_loss / (L - 1)


#%% Experiment 1: Loss Landscape
print("\n[1/4] Loss Landscape (Identifiability)...")

ensembles, times = generate_ensemble_fast(N=5, M=30, L=8, T_total=0.5)
a_range = np.linspace(0.2, 2.0, 40)
losses = [compute_loss_fast(a, ensembles, times) for a in tqdm(a_range, desc="Loss scan")]
a_opt = a_range[np.argmin(losses)]

print(f"  Optimal a = {a_opt:.3f} (true = 1.0)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(a_range, losses, 'b-', linewidth=2)
ax.axvline(x=1.0, color='g', linestyle='--', linewidth=2, label='True $a^*=1$')
ax.axvline(x=a_opt, color='r', linestyle=':', linewidth=2, label=f'$\\hat{{a}}={a_opt:.2f}$')
ax.set_xlabel('Parameter $a$', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Loss Landscape: Unique Minimum (Identifiability)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/loss_landscape.png', dpi=150)
print("  Saved: loss_landscape.png")


#%% Experiment 2: Consistency
print("\n[2/4] Consistency (Error vs n)...")

M_values = [10, 20, 50, 100, 200]
errors = []

for M in tqdm(M_values, desc="Consistency"):
    np.random.seed(42)
    ensembles, times = generate_ensemble_fast(N=5, M=M, L=8, T_total=0.5)
    a_range = np.linspace(0.6, 1.4, 25)
    losses = [compute_loss_fast(a, ensembles, times) for a in a_range]
    best_a = a_range[np.argmin(losses)]
    errors.append(abs(best_a - 1.0))

n_values = [M * 8 for M in M_values]
log_n, log_err = np.log(n_values), np.log([e+1e-6 for e in errors])
slope, intercept = np.polyfit(log_n, log_err, 1)

print(f"  Empirical rate: Error ~ n^{slope:.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(n_values, errors, 'bo-', markersize=10, linewidth=2, label='Empirical')
fitted = np.exp(intercept) * np.array(n_values)**slope
ax.loglog(n_values, fitted, 'r--', linewidth=2, label=f'$O(n^{{{slope:.2f}}})$')
ax.set_xlabel('Sample size $n$', fontsize=14)
ax.set_ylabel('Error', fontsize=14)
ax.set_title('Consistency: Error Decreases with $n$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/consistency.png', dpi=150)
print("  Saved: consistency.png")


#%% Experiment 3: Convergence Rate
print("\n[3/4] Convergence Rate...")

M_values = [20, 40, 80, 160]
mse_list = []

for M in tqdm(M_values, desc="Rate"):
    trial_errors = []
    for trial in range(5):
        np.random.seed(trial * 100 + M)
        ensembles, times = generate_ensemble_fast(N=5, M=M, L=5, T_total=0.3)
        a_range = np.linspace(0.7, 1.3, 20)
        losses = [compute_loss_fast(a, ensembles, times) for a in a_range]
        best_a = a_range[np.argmin(losses)]
        trial_errors.append((best_a - 1.0)**2)
    mse_list.append(np.mean(trial_errors))

n_values = [M * 5 for M in M_values]
log_n, log_mse = np.log(n_values), np.log(mse_list)
slope, _ = np.polyfit(log_n, log_mse, 1)

print(f"  Empirical: MSE ~ n^{slope:.2f} (theory: n^-1)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(n_values, mse_list, 'bo-', markersize=10, linewidth=2, label='Empirical')
theory = mse_list[0] * (np.array(n_values) / n_values[0])**(-1)
ax.loglog(n_values, theory, 'r--', linewidth=2, label='Theory $O(n^{-1})$')
ax.set_xlabel('Sample size $n$', fontsize=14)
ax.set_ylabel('MSE', fontsize=14)
ax.set_title(f'Convergence Rate: Empirical $n^{{{slope:.2f}}}$ vs Theory $n^{{-1}}$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/convergence_rate.png', dpi=150)
print("  Saved: convergence_rate.png")


#%% Experiment 4: Coercivity
print("\n[4/4] Coercivity Constants...")

theory_c = {1: 0.4836, 2: 0.8731, 3: 0.7339}
empirical_c = {}

for d in [1, 2, 3]:
    N, M = 8, 1000
    numerators, denominators = [], []
    
    for _ in range(M):
        X = np.random.randn(N, d)
        for i in range(N):
            sum_grad = np.zeros(d)
            for j in range(N):
                if i != j:
                    r = X[i] - X[j]
                    r_norm = np.linalg.norm(r) + 1e-10
                    grad = -np.exp(-r_norm) * r / r_norm
                    sum_grad += grad
                    denominators.append(np.sum(grad**2))
            numerators.append(np.sum(sum_grad**2))
    
    c_emp = np.mean(numerators) / (np.mean(denominators) * (N-1))
    empirical_c[d] = c_emp
    print(f"  d={d}: empirical={c_emp:.3f}, theory≥{theory_c[d]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
emp = [empirical_c[d] for d in [1,2,3]]
the = [theory_c[d] for d in [1,2,3]]
ax.bar(x - 0.15, emp, 0.3, label='Empirical', color='steelblue', alpha=0.8)
ax.bar(x + 0.15, the, 0.3, label='Theory (lower bound)', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['d=1', 'd=2', 'd=3'])
ax.set_ylabel('Coercivity $c_H$', fontsize=14)
ax.set_title('Coercivity Constants: Empirical vs Theory', fontsize=14)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.4)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/coercivity.png', dpi=150)
print("  Saved: coercivity.png")


#%% Summary Figure
print("\n[Summary] Creating 2x2 figure...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# A: Loss
ax = axes[0, 0]
ax.plot(a_range, losses, 'b-', lw=2)
ax.axvline(1.0, color='r', ls='--', lw=2)
ax.set_xlabel('Parameter $a$')
ax.set_ylabel('Loss')
ax.set_title('(A) Loss Landscape')
ax.grid(True, alpha=0.3)

# B: Consistency
ax = axes[0, 1]
ax.loglog([M*8 for M in M_values[:5]], errors, 'bo-', lw=2)
ax.set_xlabel('Sample size $n$')
ax.set_ylabel('Error')
ax.set_title('(B) Consistency')
ax.grid(True, alpha=0.3, which='both')

# C: Rate
ax = axes[1, 0]
ax.loglog(n_values, mse_list, 'bo-', lw=2, label='Empirical')
ax.loglog(n_values, theory, 'r--', lw=2, label='Theory')
ax.set_xlabel('Sample size $n$')
ax.set_ylabel('MSE')
ax.set_title('(C) Convergence Rate')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# D: Coercivity
ax = axes[1, 1]
ax.bar(x - 0.15, emp, 0.3, label='Empirical', color='steelblue')
ax.bar(x + 0.15, the, 0.3, label='Theory', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(['d=1', 'd=2', 'd=3'])
ax.set_ylabel('$c_H$')
ax.set_title('(D) Coercivity')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/summary.png', dpi=200)
plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/summary.pdf')
print("  Saved: summary.png/pdf")


#%% Save results
results = {
    'loss_landscape': {'a_opt': float(a_opt), 'a_true': 1.0},
    'consistency': {'n_values': n_values[:5], 'errors': errors, 'rate': float(slope)},
    'convergence_rate': {'n_values': n_values, 'mse': mse_list, 'rate': float(slope)},
    'coercivity': {str(d): {'empirical': float(empirical_c[d]), 'theory': theory_c[d]} for d in [1,2,3]}
}

with open('/home/swei20/ips_unlabeled_learning/experiments/results/theory_verification.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ All experiments completed!")
print("="*60)
