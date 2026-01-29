# Theoretical Analysis for LED_ips_nn

## Overview

This directory contains the theoretical analysis for the paper "Learning from Unlabeled Data for Interacting Particle Systems".

## Main Results

### 1. Loss Function Derivation (Proposition 1)
**Result:** The trajectory-free loss function is derived from energy dissipation principles:
```
E(Φ,V) = Dissipation + Diffusion - 2×(Energy change)
```
**Key insight:** At the true parameters, this equals the energy dissipation rate. For any other parameters, there's a non-negative residual.

### 2. Identifiability (Theorem 1)
**Result:** Under the coercivity condition with constant c_H > 0, the potentials (Φ*, V*) are identifiable up to additive constants.

**Proof structure:**
1. Define identifiability (up to constants)
2. Show loss minimizer implies zero residual
3. Coercivity gives: residual ≥ c_H × distance²
4. Therefore distance = 0

### 3. Coercivity Conditions (Proposition 2)
**Result:** Sufficient conditions for coercivity:
- Particles are exchangeable
- Conditional independence of differences given one particle
- Marginal variance condition on gradients

**Gaussian case bounds:**
| Dimension | c_H lower bound |
|-----------|-----------------|
| d = 1     | 0.4836         |
| d = 2     | 0.8731         |
| d = 3     | 0.7339         |

### 4. Consistency (Theorem 2)
**Result:** As n = M×L → ∞:
```
‖∇Φ̂_n - ∇Φ*‖ + ‖∇V̂_n - ∇V*‖ → 0 in probability
```

### 5. Convergence Rate (Theorem 3)
**Result:** For hypothesis spaces with finite complexity:
```
E[‖∇Φ̂ - ∇Φ*‖²] ≤ C × dim(H) / (c_H² × n)
```

### 6. Minimax Lower Bound (Theorem 4)
**Result:** For Hölder-s function class:
```
inf sup E[error²] ≥ c × n^{-2(s-1)/(2s+d)}
```
This shows the rate in Theorem 3 is optimal (up to log factors) for s=2.

### 7. Neural Network Bounds (Theorem 5)
**Result:** Total error decomposes as:
```
Error ≤ Approximation + Estimation + Discretization
     = O(W^{-2(s-1)/d}) + O(WD/n) + O(Δt)
```
Optimal network width: W ~ n^{d/(2s+d-2)}

## File Structure

```
theory/
├── README.md                    # This file
├── theoretical_analysis.tex     # Main theory section (LaTeX)
├── appendix_proofs.tex          # Detailed proofs (LaTeX)
├── verify_coercivity.py         # Numerical verification
└── figures/                     # (TODO) Diagrams
```

## How to Include in Paper

Add to `LED_ips_nn.tex`:
```latex
\input{theory/theoretical_analysis}

% In appendix:
\input{theory/appendix_proofs}
```

## Key References

1. Lu et al. (2024) "Interacting Particle Systems on Networks" - coercivity framework
2. Van der Vaart (2000) "Asymptotic Statistics" - M-estimation theory
3. Yarotsky (2017) - Neural network approximation theory
4. Wainwright (2019) "High-Dimensional Statistics" - minimax theory

## TODO

- [ ] Add figures for proof intuition
- [ ] Extend to non-Gaussian initial distributions
- [ ] Analyze adaptive/regularized estimators
- [ ] Computational complexity analysis
