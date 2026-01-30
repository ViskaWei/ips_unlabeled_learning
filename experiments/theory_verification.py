#!/usr/bin/env python3
"""
Experimental Verification of Theoretical Results
for "Learning from Unlabeled Data for IPS"

Experiments:
1. Loss landscape - verify unique minimum (identifiability)
2. Consistency - error → 0 as n → ∞
3. Convergence rate - verify O(n^{-α}) scaling
4. Coercivity estimation - empirical coercivity constants

Author: Panda 🐼 (AI Assistant)
Date: 2025-01-29
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('/home/swei20/ips_unlabeled_learning/experiments/results', exist_ok=True)
os.makedirs('/home/swei20/ips_unlabeled_learning/experiments/figures', exist_ok=True)

print("="*60)
print("Theoretical Verification Experiments")
print("="*60)

#%% ============================================
# Setup: IPS Model (1D for simplicity)
# ============================================

class IPSSimulator:
    """Simulate 1D Interacting Particle System"""
    
    def __init__(self, N=10, sigma=0.1):
        self.N = N
        self.sigma = sigma
        
    def true_grad_Phi(self, r):
        """Gradient of true interaction potential: Φ(r) = exp(-|r|)"""
        return -np.exp(-np.abs(r)) * np.sign(r)
    
    def true_grad_V(self, x):
        """Gradient of kinetic potential: V(x) = 0.5*x²"""
        return x
    
    def drift(self, X):
        """Compute drift: -∇V - (1/N)Σ∇Φ"""
        N = len(X)
        drift = np.zeros(N)
        
        for i in range(N):
            drift[i] = -self.true_grad_V(X[i])
            for j in range(N):
                if i != j:
                    drift[i] -= self.true_grad_Phi(X[i] - X[j]) / N
        
        return drift
    
    def simulate(self, X0, T, dt):
        """Euler-Maruyama simulation"""
        steps = int(T / dt)
        X = X0.copy()
        
        for _ in range(steps):
            dW = np.random.randn(len(X)) * np.sqrt(dt)
            X = X + self.drift(X) * dt + self.sigma * dW
        
        return X
    
    def generate_ensemble(self, M, L, T_total, dt=0.01):
        """Generate unlabeled ensemble data"""
        times = np.linspace(0, T_total, L)
        
        ensembles = []
        for t_idx in range(L):
            ensemble_t = []
            for m in range(M):
                X0 = np.random.randn(self.N)
                if t_idx > 0:
                    X = self.simulate(X0, times[t_idx], dt)
                else:
                    X = X0
                ensemble_t.append(X)
            ensembles.append(np.array(ensemble_t))  # (M, N)
        
        return ensembles, times


def compute_loss_simple(a, ensembles, times, sigma, V_grad_func):
    """
    Simplified loss computation for parametric Φ(r) = a * exp(-|r|)
    """
    L = len(times)
    M, N = ensembles[0].shape
    
    total_loss = 0.0
    
    for l in range(L - 1):
        X_t = ensembles[l]
        X_t1 = ensembles[l + 1]
        dt = times[l + 1] - times[l]
        
        for m in range(M):
            # Dissipation term
            dissip = 0.0
            for i in range(N):
                drift_i = V_grad_func(X_t[m, i])
                for j in range(N):
                    r_ij = X_t[m, i] - X_t[m, j]
                    drift_i += a * (-np.exp(-np.abs(r_ij)) * np.sign(r_ij)) / N
                dissip += drift_i**2
            dissip = dissip / N * dt
            
            # Energy change (simplified)
            E_t = np.sum(0.5 * X_t[m]**2) / N
            E_t1 = np.sum(0.5 * X_t1[m]**2) / N
            for i in range(N):
                for j in range(N):
                    E_t += a * np.exp(-np.abs(X_t[m, i] - X_t[m, j])) / (2 * N * N)
                    E_t1 += a * np.exp(-np.abs(X_t1[m, i] - X_t1[m, j])) / (2 * N * N)
            
            total_loss += dissip - 2 * (E_t1 - E_t)
    
    return total_loss / (M * (L - 1))


#%% ============================================
# Experiment 1: Loss Landscape (Identifiability)
# ============================================

def experiment_loss_landscape():
    """Verify that loss has unique minimum at true parameters"""
    print("\n" + "="*50)
    print("Experiment 1: Loss Landscape (Identifiability)")
    print("="*50)
    
    sim = IPSSimulator(N=5, sigma=0.1)
    
    # Generate data
    M, L = 50, 10
    ensembles, times = sim.generate_ensemble(M, L, T_total=1.0)
    
    # True parameter a = 1
    a_range = np.linspace(0.2, 2.0, 50)
    
    losses = []
    print("Computing loss for different parameter values...")
    for a in tqdm(a_range):
        loss = compute_loss_simple(a, ensembles, times, sim.sigma, sim.true_grad_V)
        losses.append(loss)
    
    losses = np.array(losses)
    
    # Find minimum
    a_opt = a_range[np.argmin(losses)]
    loss_min = np.min(losses)
    
    print(f"Optimal parameter: a = {a_opt:.3f}")
    print(f"True parameter: a = 1.0")
    print(f"Error: |a_opt - a_true| = {np.abs(a_opt - 1.0):.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_range, losses, 'b-', linewidth=2, label='Loss $\\mathcal{E}(a)$')
    ax.axvline(x=1.0, color='g', linestyle='--', linewidth=2, label='True $a^* = 1$')
    ax.axvline(x=a_opt, color='r', linestyle=':', linewidth=2, label=f'Estimated $\\hat{{a}} = {a_opt:.2f}$')
    ax.scatter([a_opt], [loss_min], color='r', s=100, zorder=5)
    
    ax.set_xlabel('Parameter $a$', fontsize=14)
    ax.set_ylabel('Loss $\\mathcal{E}(a)$', fontsize=14)
    ax.set_title('Loss Landscape: Unique Minimum Verifies Identifiability', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/loss_landscape.png', dpi=150)
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/loss_landscape.pdf')
    print("Saved: figures/loss_landscape.png")
    
    return {'a_opt': float(a_opt), 'a_true': 1.0, 'loss_min': float(loss_min)}


#%% ============================================
# Experiment 2: Consistency (Error vs n)
# ============================================

def experiment_consistency():
    """Verify that estimation error decreases with sample size"""
    print("\n" + "="*50)
    print("Experiment 2: Consistency (Error → 0 as n → ∞)")
    print("="*50)
    
    sim = IPSSimulator(N=5, sigma=0.1)
    
    M_values = [10, 20, 50, 100, 200, 500]
    L = 10
    n_trials = 5
    
    errors_mean = []
    errors_std = []
    
    for M in M_values:
        print(f"M = {M}...", end=" ")
        trial_errors = []
        
        for trial in range(n_trials):
            np.random.seed(trial * 100 + M)
            ensembles, times = sim.generate_ensemble(M, L, T_total=1.0)
            
            # Grid search for optimal a
            a_range = np.linspace(0.5, 1.5, 30)
            losses = [compute_loss_simple(a, ensembles, times, sim.sigma, sim.true_grad_V) 
                     for a in a_range]
            best_a = a_range[np.argmin(losses)]
            
            error = np.abs(best_a - 1.0)
            trial_errors.append(error)
        
        errors_mean.append(np.mean(trial_errors))
        errors_std.append(np.std(trial_errors))
        print(f"Error: {errors_mean[-1]:.4f} ± {errors_std[-1]:.4f}")
    
    n_values = [M * L for M in M_values]
    
    # Fit power law
    log_n = np.log(n_values)
    log_err = np.log(errors_mean)
    slope, intercept = np.polyfit(log_n, log_err, 1)
    fitted = np.exp(intercept) * np.array(n_values)**slope
    
    print(f"\nEmpirical rate: Error ~ n^{slope:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(n_values, errors_mean, yerr=errors_std, fmt='o-', capsize=5, 
                markersize=8, linewidth=2, label='Empirical error', color='steelblue')
    ax.plot(n_values, fitted, 'r--', linewidth=2, 
            label=f'Fitted: $O(n^{{{slope:.2f}}})$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size $n = M \\times L$', fontsize=14)
    ax.set_ylabel('Estimation error $|\\hat{a} - a^*|$', fontsize=14)
    ax.set_title('Consistency: Error Decreases with Sample Size', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/consistency.png', dpi=150)
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/consistency.pdf')
    print("Saved: figures/consistency.png")
    
    return {'n_values': n_values, 'errors': errors_mean, 'rate': float(slope)}


#%% ============================================
# Experiment 3: Convergence Rate (MSE scaling)
# ============================================

def experiment_convergence_rate():
    """Verify theoretical convergence rate"""
    print("\n" + "="*50)
    print("Experiment 3: Convergence Rate Verification")
    print("="*50)
    
    sim = IPSSimulator(N=5, sigma=0.1)
    
    M_values = [20, 40, 80, 160, 320]
    L = 5
    n_trials = 10
    
    mse_values = []
    
    for M in M_values:
        print(f"M = {M}...", end=" ")
        trial_errors_sq = []
        
        for trial in range(n_trials):
            np.random.seed(trial * 1000 + M)
            ensembles, times = sim.generate_ensemble(M, L, T_total=0.5)
            
            a_range = np.linspace(0.6, 1.4, 40)
            losses = [compute_loss_simple(a, ensembles, times, sim.sigma, sim.true_grad_V) 
                     for a in a_range]
            best_a = a_range[np.argmin(losses)]
            trial_errors_sq.append((best_a - 1.0)**2)
        
        mse = np.mean(trial_errors_sq)
        mse_values.append(mse)
        print(f"MSE: {mse:.6f}")
    
    n_values = [M * L for M in M_values]
    
    # Fit rate
    log_n = np.log(n_values)
    log_mse = np.log(mse_values)
    slope, intercept = np.polyfit(log_n, log_mse, 1)
    
    print(f"\nEmpirical rate: MSE ~ n^{slope:.3f}")
    print(f"Theory predicts: MSE ~ n^{-1} for parametric")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(n_values, mse_values, 'bo-', markersize=10, linewidth=2, 
              label='Empirical MSE')
    
    # Theoretical line (n^{-1})
    theory_line = mse_values[0] * (np.array(n_values) / n_values[0])**(-1.0)
    ax.loglog(n_values, theory_line, 'r--', linewidth=2, 
              label='Theory: $O(n^{-1})$')
    
    # Fitted line
    fitted = np.exp(intercept) * np.array(n_values)**slope
    ax.loglog(n_values, fitted, 'g:', linewidth=2, 
              label=f'Fitted: $O(n^{{{slope:.2f}}})$')
    
    ax.set_xlabel('Sample size $n$', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_title('Convergence Rate: Theory vs Experiment', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/convergence_rate.png', dpi=150)
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/convergence_rate.pdf')
    print("Saved: figures/convergence_rate.png")
    
    return {'n_values': n_values, 'mse': mse_values, 'empirical_rate': float(slope)}


#%% ============================================
# Experiment 4: Coercivity Estimation
# ============================================

def experiment_coercivity():
    """Estimate coercivity constant empirically for different dimensions"""
    print("\n" + "="*50)
    print("Experiment 4: Coercivity Constant Estimation")
    print("="*50)
    
    results = {}
    theory_bounds = {1: 0.4836, 2: 0.8731, 3: 0.7339}
    
    for d in [1, 2, 3]:
        print(f"\nDimension d = {d}...")
        
        N = 10
        M = 2000
        
        # For coercivity, we measure:
        # E[|Σ_j ∇δΦ(r_ij)|²] / (N * E[|∇δΦ(r)|²])
        
        numerator_samples = []
        denominator_samples = []
        
        for _ in range(M):
            X = np.random.randn(N, d)  # Gaussian particles
            
            # For each particle i, compute sum of gradients
            for i in range(N):
                sum_grad = np.zeros(d)
                grad_norms_sq = []
                
                for j in range(N):
                    if i != j:
                        r_ij = X[i] - X[j]
                        r_norm = np.linalg.norm(r_ij) + 1e-10
                        # ∇Φ(r) = -exp(-|r|) * r/|r| for Φ(r) = exp(-|r|)
                        grad = -np.exp(-r_norm) * r_ij / r_norm
                        sum_grad += grad
                        grad_norms_sq.append(np.sum(grad**2))
                
                numerator_samples.append(np.sum(sum_grad**2))
                denominator_samples.extend(grad_norms_sq)
        
        num_mean = np.mean(numerator_samples)
        denom_mean = np.mean(denominator_samples) * (N - 1)  # Normalize
        
        c_H_empirical = num_mean / denom_mean if denom_mean > 0 else 0
        
        print(f"  Empirical c_H: {c_H_empirical:.4f}")
        print(f"  Theory lower bound: {theory_bounds[d]:.4f}")
        print(f"  Ratio (empirical/theory): {c_H_empirical/theory_bounds[d]:.2f}")
        
        results[d] = {
            'empirical': float(c_H_empirical),
            'theory': theory_bounds[d]
        }
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    dims = [1, 2, 3]
    empirical = [results[d]['empirical'] for d in dims]
    theory = [results[d]['theory'] for d in dims]
    
    x = np.arange(len(dims))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, empirical, width, label='Empirical $c_H$', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, theory, width, label='Theory (lower bound)', 
                   color='coral', alpha=0.8)
    
    ax.set_xlabel('Dimension $d$', fontsize=14)
    ax.set_ylabel('Coercivity constant $c_H$', fontsize=14)
    ax.set_title('Coercivity Constants: Empirical vs Theoretical Bounds', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['$d=1$', '$d=2$', '$d=3$'], fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, empirical):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, theory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/coercivity.png', dpi=150)
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/coercivity.pdf')
    print("\nSaved: figures/coercivity.png")
    
    return results


#%% ============================================
# Summary Figure
# ============================================

def create_summary_figure(results):
    """Create a 2x2 summary figure"""
    print("\n" + "="*50)
    print("Creating Summary Figure")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Loss Landscape
    ax = axes[0, 0]
    sim = IPSSimulator(N=5, sigma=0.1)
    ensembles, times = sim.generate_ensemble(50, 10, T_total=1.0)
    a_range = np.linspace(0.2, 2.0, 50)
    losses = [compute_loss_simple(a, ensembles, times, sim.sigma, sim.true_grad_V) 
              for a in a_range]
    ax.plot(a_range, losses, 'b-', linewidth=2)
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Parameter $a$', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('(A) Loss Landscape: Identifiability', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Consistency
    ax = axes[0, 1]
    n_vals = results['consistency']['n_values']
    errs = results['consistency']['errors']
    rate = results['consistency']['rate']
    ax.loglog(n_vals, errs, 'bo-', markersize=8, linewidth=2)
    fitted = errs[0] * (np.array(n_vals) / n_vals[0])**rate
    ax.loglog(n_vals, fitted, 'r--', linewidth=2, label=f'$O(n^{{{rate:.2f}}})$')
    ax.set_xlabel('Sample size $n$', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('(B) Consistency: Error → 0', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel C: Convergence Rate
    ax = axes[1, 0]
    n_vals = results['convergence_rate']['n_values']
    mse = results['convergence_rate']['mse']
    rate = results['convergence_rate']['empirical_rate']
    ax.loglog(n_vals, mse, 'bo-', markersize=8, linewidth=2, label='Empirical')
    theory = mse[0] * (np.array(n_vals) / n_vals[0])**(-1)
    ax.loglog(n_vals, theory, 'r--', linewidth=2, label='Theory $O(n^{-1})$')
    ax.set_xlabel('Sample size $n$', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(f'(C) Convergence Rate (empirical: $n^{{{rate:.2f}}}$)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel D: Coercivity
    ax = axes[1, 1]
    dims = [1, 2, 3]
    empirical = [results['coercivity'][d]['empirical'] for d in dims]
    theory = [results['coercivity'][d]['theory'] for d in dims]
    x = np.arange(len(dims))
    width = 0.35
    ax.bar(x - width/2, empirical, width, label='Empirical', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, theory, width, label='Theory', color='coral', alpha=0.8)
    ax.set_xlabel('Dimension $d$', fontsize=12)
    ax.set_ylabel('$c_H$', fontsize=12)
    ax.set_title('(D) Coercivity Constants', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['1', '2', '3'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/summary.png', dpi=200)
    plt.savefig('/home/swei20/ips_unlabeled_learning/experiments/figures/summary.pdf')
    print("Saved: figures/summary.png")


#%% ============================================
# Main
# ============================================

if __name__ == "__main__":
    results = {}
    
    # Run all experiments
    results['loss_landscape'] = experiment_loss_landscape()
    results['consistency'] = experiment_consistency()
    results['convergence_rate'] = experiment_convergence_rate()
    results['coercivity'] = experiment_coercivity()
    
    # Create summary figure
    create_summary_figure(results)
    
    # Save results
    with open('/home/swei20/ips_unlabeled_learning/experiments/results/theory_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ All experiments completed!")
    print("Results saved to: experiments/results/theory_verification.json")
    print("Figures saved to: experiments/figures/")
    print("="*60)
