"""
Complete verification of Gaussian coercivity integrals using SymPy
"""
import sympy as sp
from sympy import pi, sqrt, exp, integrate, oo, simplify, Rational, symbols
from sympy import cos, sin, Abs
import numpy as np
from scipy import integrate as scipy_integrate

print("=" * 70)
print("Gaussian Coercivity Integral Verification")
print("=" * 70)

# ============================================================
# DIMENSION d = 1
# ============================================================
print("\n" + "=" * 70)
print("DIMENSION d = 1")
print("=" * 70)

r, s = symbols('r s', real=True)

# The key quantity is:
# E[<∇Φ(r_12), ∇Φ(r_13)>] where r_12 = X_2 - X_1, r_13 = X_3 - X_1
# For X_i ~ N(0, 1) iid:
#   r_12 ~ N(0, 2)
#   r_13 ~ N(0, 2)
#   Cov(r_12, r_13) = Var(X_1) = 1
# So (r_12, r_13) ~ N(0, [[2, 1], [1, 2]])

print("\nJoint distribution of (r_12, r_13):")
print("  (r_12, r_13) ~ N(0, Σ) where Σ = [[2, 1], [1, 2]]")
print("  det(Σ) = 4 - 1 = 3")
print("  Σ^{-1} = (1/3) * [[2, -1], [-1, 2]]")

# PDF: p(r,s) = (1/(2π√3)) exp(-(2r² - 2rs + 2s²)/(2*3))
#            = (1/(2π√3)) exp(-(r² - rs + s²)/3)

print("\nPDF: p(r,s) = (1/(2π√3)) exp(-(r² - rs + s²)/3)")

# For d=1, ∇Φ(r) = Φ'(r) * sign(r) (scalar)
# E[Φ'(r_12) * Φ'(r_13) * sign(r_12) * sign(r_13)]

# The coercivity constant comes from bounding:
# E[|Φ'(r_12) + Φ'(r_13)|²] >= c * (E[Φ'(r_12)²] + E[Φ'(r_13)²])

# Expanding: E[Φ'(r_12)²] + 2*E[Φ'(r_12)*Φ'(r_13)] + E[Φ'(r_13)²]
#          = 2*E[Φ'(r)²] * (1 + ρ)
# where ρ = E[Φ'(r_12)*Φ'(r_13)] / E[Φ'(r)²]

# For worst case, we need to find sup_Φ |ρ|

print("\n--- Computing correlation bound ---")

# For radial Φ (even function): Φ(r) = φ(|r|)
# ∇Φ(r) = φ'(|r|) * sign(r)

# E[sign(r_12) * sign(r_13)] under joint Gaussian
# This is P(r_12 > 0, r_13 > 0) + P(r_12 < 0, r_13 < 0) - P(r_12 > 0, r_13 < 0) - P(r_12 < 0, r_13 > 0)

# For bivariate normal with correlation ρ_0 = 1/(√2 * √2) = 1/2:
# P(X > 0, Y > 0) = 1/4 + arcsin(ρ_0)/(2π)

rho_0 = Rational(1, 2)  # correlation = Cov/sqrt(Var1*Var2) = 1/sqrt(2*2) = 1/2
prob_same_sign = Rational(1, 4) + sp.asin(rho_0) / (2 * pi)
prob_same_sign_val = float(prob_same_sign.evalf())

print(f"Correlation between r_12 and r_13: ρ = 1/2")
print(f"P(same sign) = 1/4 + arcsin(1/2)/(2π) = 1/4 + (π/6)/(2π) = 1/4 + 1/12 = 1/3")
print(f"E[sign(r_12)*sign(r_13)] = 2*P(same sign) - 1 = 2/3 - 1 = -1/3")

# Wait, let me recalculate
# P(r_12 > 0, r_13 > 0) = 1/4 + arcsin(1/2)/(2π) = 1/4 + (π/6)/(2π) = 1/4 + 1/12 = 4/12 = 1/3
# Similarly P(r_12 < 0, r_13 < 0) = 1/3
# P(r_12 > 0, r_13 < 0) = P(r_12 < 0, r_13 > 0) = (1 - 2/3)/2 = 1/6

# E[sign(r_12)*sign(r_13)] = (1/3 + 1/3) - (1/6 + 1/6) = 2/3 - 1/3 = 1/3

print("\nCorrection:")
print("  P(++|ρ=1/2) = 1/4 + arcsin(1/2)/(2π) = 1/4 + 1/12 = 1/3")
print("  P(--|ρ=1/2) = 1/3")
print("  P(+-) = P(-+) = 1/6")
print("  E[sign(r_12)*sign(r_13)] = 1/3 + 1/3 - 1/6 - 1/6 = 1/3")

# So for constant φ'(|r|) = c:
# E[c² * sign(r_12) * sign(r_13)] = c² * 1/3
# E[c²] = c²
# ρ_worst = 1/3

print("\nFor constant derivative φ'(|r|) = c:")
print("  ρ = E[sign(r_12)*sign(r_13)] = 1/3")

# But for non-constant φ', we need to compute:
# E[φ'(|r_12|) * φ'(|r_13|) * sign(r_12) * sign(r_13)]

print("\n--- General case with SymPy ---")

# Use numerical integration for verification
def numerical_coercivity_d1():
    """Compute coercivity by numerical integration"""
    from scipy.stats import multivariate_normal
    
    # Joint distribution
    cov = np.array([[2, 1], [1, 2]])
    rv = multivariate_normal(mean=[0, 0], cov=cov)
    
    # For worst-case, consider φ'(r) = sign(r) * f(|r|) for various f
    # The correlation is maximized when f is constant
    
    # Compute E[sign(r)*sign(s)] numerically
    def integrand(x):
        r, s = x
        return np.sign(r) * np.sign(s) * rv.pdf(x)
    
    # Monte Carlo
    np.random.seed(42)
    samples = rv.rvs(size=100000)
    sign_corr = np.mean(np.sign(samples[:, 0]) * np.sign(samples[:, 1]))
    
    print(f"  Monte Carlo E[sign(r_12)*sign(r_13)] = {sign_corr:.6f}")
    print(f"  Theoretical: 1/3 = {1/3:.6f}")
    
    # The coercivity constant
    # E[|∇Φ(r_12) + ∇Φ(r_13)|²] = E[Φ'²](2 + 2*ρ)
    # where ρ = 1/3 for worst case
    # >= E[Φ'²] * 2 * (1 + 1/3) = E[Φ'²] * 8/3
    # But we need >= c_H * 2 * E[Φ'²]
    # So c_H >= (1 + 1/3)/2... wait this is wrong
    
    # Actually: E[|A + B|²] = E[A²] + 2E[AB] + E[B²] = 2E[A²](1 + ρ)
    # For coercivity: E[|A + B|²] >= c_H * (E[A²] + E[B²]) = c_H * 2E[A²]
    # So we need 2(1 + ρ) >= 2c_H, i.e., c_H <= 1 + ρ
    # But ρ can be negative! Worst case is ρ = 1/3, so c_H = 1 + 1/3 = 4/3 > 1
    
    # Hmm, this seems to give c_H > 1 which means the bound is trivial
    # Let me re-read the original paper's definition
    
    return sign_corr

sign_corr = numerical_coercivity_d1()

print("\n--- Re-examining the coercivity definition ---")
print("The paper defines coercivity as:")
print("  E[|∇δV + ∇δΦ*μ|²] >= c_H * (||∇δV||² + ||∇δΦ||²)")
print("\nFor N=2 particles, this becomes:")
print("  E[|∇δV(X_1) + ∇δΦ(X_2-X_1)|²] >= c_H * (||∇δV||² + ||∇δΦ||²)")
print("\nThe issue is correlation between different terms.")

# ============================================================
# DIMENSION d = 2, 3
# ============================================================
print("\n" + "=" * 70)
print("DIMENSION d = 2, 3 (Numerical)")
print("=" * 70)

def compute_coercivity_nd(d, n_samples=100000):
    """Compute coercivity constant for d dimensions"""
    np.random.seed(42)
    
    # Generate iid Gaussian particles
    X1 = np.random.randn(n_samples, d)
    X2 = np.random.randn(n_samples, d)
    X3 = np.random.randn(n_samples, d)
    
    r12 = X2 - X1  # shape (n_samples, d)
    r13 = X3 - X1
    
    # For radial Φ: ∇Φ(r) = φ'(|r|) * r/|r|
    # Consider φ'(r) = 1 (constant)
    
    norm_r12 = np.linalg.norm(r12, axis=1, keepdims=True)
    norm_r13 = np.linalg.norm(r13, axis=1, keepdims=True)
    
    # Unit vectors
    u12 = r12 / (norm_r12 + 1e-10)
    u13 = r13 / (norm_r13 + 1e-10)
    
    # E[<u12, u13>] = E[cos(angle)]
    cos_angle = np.sum(u12 * u13, axis=1)
    rho = np.mean(cos_angle)
    
    # Coercivity: E[|u12 + u13|²] = E[2 + 2<u12,u13>] = 2(1 + ρ)
    # We need E[|u12 + u13|²] >= c_H * (E[|u12|²] + E[|u13|²]) = c_H * 2
    # So c_H <= 1 + ρ
    
    # But the paper reports c_H ~ 0.5 for d=1, which suggests ρ ~ -0.5?
    
    print(f"d={d}: E[cos(angle)] = {rho:.6f}")
    print(f"      c_H upper bound = 1 + ρ = {1 + rho:.6f}")
    
    return rho

for d in [1, 2, 3]:
    compute_coercivity_nd(d)

print("\n" + "=" * 70)
print("ISSUE IDENTIFIED")
print("=" * 70)
print("""
The coercivity constants in the paper (c_H ~ 0.48 for d=1) suggest that
the correlation ρ ≈ -0.52, which would mean E[cos(angle)] < 0.

But our calculation shows E[cos(angle)] ≈ 0.33 for d=1.

POSSIBLE EXPLANATIONS:
1. The paper uses a different normalization
2. The paper considers the VARIANCE, not just the expectation
3. The definition involves the stationary distribution, not iid

Let me check the variance-based definition...
""")

def check_variance_coercivity(d, n_samples=100000):
    """Check if coercivity involves conditional variance"""
    np.random.seed(42)
    
    X1 = np.random.randn(n_samples, d)
    X2 = np.random.randn(n_samples, d)
    X3 = np.random.randn(n_samples, d)
    
    r12 = X2 - X1
    r13 = X3 - X1
    
    norm_r12 = np.linalg.norm(r12, axis=1, keepdims=True)
    norm_r13 = np.linalg.norm(r13, axis=1, keepdims=True)
    
    u12 = r12 / (norm_r12 + 1e-10)
    u13 = r13 / (norm_r13 + 1e-10)
    
    # The Fei Lu definition (from equation 3.15 in their paper):
    # c_H = inf_{φ} E[|Σ_j ∇φ(r_{1j})|²] / ((N-1) * E[|∇φ(r)|²])
    
    # For N=3, this is:
    # E[|∇φ(r_12) + ∇φ(r_13)|²] / (2 * E[|∇φ(r)|²])
    
    # For constant |∇φ| = 1:
    # E[|u12 + u13|²] / 2 = E[2 + 2<u12,u13>] / 2 = 1 + ρ
    
    sum_vec = u12 + u13
    E_sum_sq = np.mean(np.sum(sum_vec**2, axis=1))
    E_single_sq = 1.0  # |u|² = 1
    
    c_H = E_sum_sq / (2 * E_single_sq)
    
    print(f"d={d}: E[|u12+u13|²] = {E_sum_sq:.6f}")
    print(f"      c_H = E[|u12+u13|²] / 2 = {c_H:.6f}")
    
    # The paper's c_H ≈ 0.48 for d=1 doesn't match this
    # Unless they're computing something different...
    
    return c_H

print("\n--- Checking the actual ratio ---")
for d in [1, 2, 3]:
    check_variance_coercivity(d)

print("\n" + "=" * 70)
print("CONCLUSION")  
print("=" * 70)
print("""
The numerical check gives c_H ≈ 1.33 for d=1, NOT 0.48.

This suggests the paper's formula or our interpretation is wrong.

Possible issues:
1. The I(d, G_d) formula in the paper may have errors
2. The definition of coercivity may be different
3. Need to check Fei Lu's original paper for the exact definition
""")
