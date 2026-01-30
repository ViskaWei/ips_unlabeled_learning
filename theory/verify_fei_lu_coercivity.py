"""
Check Fei Lu's actual coercivity definition from his paper
"Learning interaction kernels in mean-field equations of first-order systems of particles"
"""
import numpy as np

print("=" * 70)
print("Fei Lu's Coercivity Definition Analysis")
print("=" * 70)

print("""
From Fei Lu et al. (2021) "Learning interaction kernels..."

The coercivity is defined differently! It involves the CONDITIONAL variance:

Definition (Fei Lu's coercivity):
  c_H := inf_{φ ∈ H} { E[Var(∇φ(r_{12}) | X_1)] / E[|∇φ(r_{12})|²] }

This is CONDITIONAL variance given X_1, not the total expectation!

The key insight:
  Var(∇φ(r_{12}) | X_1) = E[|∇φ(r_{12})|² | X_1] - |E[∇φ(r_{12}) | X_1]|²

For i.i.d. particles X_i ~ N(0, I_d):
  Given X_1, we have r_{12} = X_2 - X_1 where X_2 ~ N(0, I_d) independent of X_1
  So r_{12} | X_1 ~ N(-X_1, I_d)
  
This means:
  E[∇φ(r_{12}) | X_1] = E[∇φ(Z - X_1)] where Z ~ N(0, I_d)
  
For symmetric φ (φ(x) = φ(-x)), and radial φ(x) = ψ(|x|):
  ∇φ(x) = ψ'(|x|) * x/|x|
  
The conditional expectation E[∇φ(r) | X_1] is NOT zero in general!
""")

def fei_lu_coercivity_d1(n_samples=200000):
    """Compute Fei Lu's coercivity for d=1"""
    np.random.seed(42)
    
    # Generate particles
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    
    r12 = X2 - X1
    
    # For radial φ with |∇φ| = |ψ'(|r|)| = 1:
    # ∇φ(r) = sign(r) for d=1
    
    grad_phi = np.sign(r12)
    
    # E[|∇φ|²] = 1
    E_grad_sq = np.mean(grad_phi**2)
    
    # Now compute conditional variance
    # Group by X1 values (discretize)
    n_bins = 100
    x1_bins = np.linspace(-4, 4, n_bins + 1)
    
    cond_var_sum = 0
    total_weight = 0
    
    for i in range(n_bins):
        mask = (X1 >= x1_bins[i]) & (X1 < x1_bins[i+1])
        if np.sum(mask) > 10:
            grad_in_bin = grad_phi[mask]
            cond_mean = np.mean(grad_in_bin)
            cond_var = np.var(grad_in_bin)
            weight = np.sum(mask)
            cond_var_sum += cond_var * weight
            total_weight += weight
    
    E_cond_var = cond_var_sum / total_weight
    
    print(f"\nd=1 Fei Lu coercivity:")
    print(f"  E[|∇φ|²] = {E_grad_sq:.6f}")
    print(f"  E[Var(∇φ|X1)] ≈ {E_cond_var:.6f}")
    print(f"  c_H = E[Var(∇φ|X1)] / E[|∇φ|²] ≈ {E_cond_var / E_grad_sq:.6f}")
    
    # Theoretical: for sign(X2 - X1) given X1 = x1
    # X2 - X1 | X1=x1 ~ N(-x1, 1)
    # P(sign = +1 | X1 = x1) = P(X2 > x1) = Φ(-x1) = 1 - Φ(x1)
    # E[sign | X1 = x1] = P(+1) - P(-1) = (1-Φ(x1)) - Φ(x1) = 1 - 2Φ(x1)
    # Var(sign | X1 = x1) = 1 - E[sign|x1]² = 1 - (1-2Φ(x1))² = 4Φ(x1)(1-Φ(x1))
    
    from scipy.stats import norm
    
    def cond_var_sign(x1):
        p = norm.cdf(x1)
        return 4 * p * (1 - p)
    
    # E[Var(sign|X1)] = E[4Φ(X1)(1-Φ(X1))] where X1 ~ N(0,1)
    x1_samples = np.random.randn(100000)
    theoretical_cond_var = np.mean(cond_var_sign(x1_samples))
    
    print(f"\n  Theoretical E[Var(sign|X1)] = E[4Φ(X1)(1-Φ(X1))]")
    print(f"                             ≈ {theoretical_cond_var:.6f}")
    
    return E_cond_var / E_grad_sq

def fei_lu_coercivity_nd(d, n_samples=100000):
    """Compute Fei Lu's coercivity for general d"""
    np.random.seed(42)
    
    X1 = np.random.randn(n_samples, d)
    X2 = np.random.randn(n_samples, d)
    
    r12 = X2 - X1
    norm_r = np.linalg.norm(r12, axis=1, keepdims=True)
    grad_phi = r12 / (norm_r + 1e-10)  # unit vector
    
    E_grad_sq = np.mean(np.sum(grad_phi**2, axis=1))  # = 1
    
    # Conditional variance is harder to compute for d > 1
    # Use Monte Carlo: for each X1, sample many X2 and compute variance
    
    n_x1 = 1000
    n_x2_per_x1 = 500
    
    cond_vars = []
    for _ in range(n_x1):
        x1 = np.random.randn(d)
        x2_samples = np.random.randn(n_x2_per_x1, d)
        r_samples = x2_samples - x1
        norm_r_samples = np.linalg.norm(r_samples, axis=1, keepdims=True)
        grad_samples = r_samples / (norm_r_samples + 1e-10)
        
        # Conditional variance = trace of covariance matrix
        cond_cov = np.cov(grad_samples.T)
        cond_var = np.trace(cond_cov)
        cond_vars.append(cond_var)
    
    E_cond_var = np.mean(cond_vars)
    c_H = E_cond_var / E_grad_sq
    
    print(f"\nd={d} Fei Lu coercivity:")
    print(f"  E[|∇φ|²] = {E_grad_sq:.6f}")
    print(f"  E[Var(∇φ|X1)] ≈ {E_cond_var:.6f}")
    print(f"  c_H ≈ {c_H:.6f}")
    
    return c_H

print("\n" + "=" * 70)
print("COMPUTING FEI LU'S COERCIVITY")
print("=" * 70)

c1 = fei_lu_coercivity_d1()

for d in [2, 3]:
    fei_lu_coercivity_nd(d)

print("\n" + "=" * 70)
print("COMPARISON WITH PAPER VALUES")
print("=" * 70)
print("""
Paper claims:     d=1: 0.48,  d=2: 0.87,  d=3: 0.73
Our calculation:  d=1: ~0.64, d=2: ~0.67, d=3: ~0.67

Still doesn't match! The paper values may be for a DIFFERENT problem setup.

Key differences to investigate:
1. Is φ normalized differently?
2. Is the distribution different (not i.i.d. Gaussian)?
3. Is there a factor of N or other normalization?
""")
