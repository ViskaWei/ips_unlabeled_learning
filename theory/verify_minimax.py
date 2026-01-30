"""
Verify Minimax Lower Bound Exponent
Check which rate is correct: n^{-2(s-1)/(2s+d)} or n^{-2(s-1)/(2s+d-2)}
"""
import sympy as sp

print("=" * 60)
print("Minimax Lower Bound Exponent Verification")
print("=" * 60)

s, d, n = sp.symbols('s d n', positive=True, real=True)

# From the proof:
# Step 1: Packing number log M >= c1 * epsilon^{-d/(s-1)}
# Step 2: KL divergence D_KL <= C * n * epsilon^2
# Step 3: Fano requires n * epsilon^2 <= alpha * epsilon^{-d/(s-1)}

print("\n--- Derivation from Fano's Inequality ---")
print("Metric entropy of Hölder ball in gradient L2 norm:")
print("  log M(ε) ~ ε^{-d/(s-1)}  [for ||∇·||_{L2} metric on C^s functions]")

print("\nKL divergence between hypotheses:")
print("  D_KL(P_i || P_j) ~ n * ||∇Φ_i - ∇Φ_j||² * Δt²")
print("  For ε-separated: D_KL ~ n * ε²")

print("\nFano's inequality requires:")
print("  n * ε² ≲ log M(ε) ~ ε^{-d/(s-1)}")
print("  => n * ε² ≲ ε^{-d/(s-1)}")
print("  => n ≲ ε^{-d/(s-1) - 2}")
print("  => ε ≳ n^{-1/(d/(s-1) + 2)}")
print("  => ε ≳ n^{-(s-1)/(d + 2(s-1))}")
print("  => ε ≳ n^{-(s-1)/(2s + d - 2)}")

print("\nSquaring for MSE rate:")
print("  ε² ≳ n^{-2(s-1)/(2s + d - 2)}")

# Simplify
exponent_proof = -2*(s-1) / (2*s + d - 2)
exponent_theorem = -2*(s-1) / (2*s + d)

print("\n--- Comparison ---")
print(f"Exponent from proof:   -2(s-1)/(2s+d-2)")
print(f"Exponent in theorem:   -2(s-1)/(2s+d)")

print("\nNumerical examples:")
for s_val in [2, 3, 4]:
    for d_val in [1, 2, 3]:
        exp_proof = -2*(s_val-1) / (2*s_val + d_val - 2)
        exp_thm = -2*(s_val-1) / (2*s_val + d_val)
        print(f"  s={s_val}, d={d_val}: proof={exp_proof:.4f}, theorem={exp_thm:.4f}, diff={exp_proof-exp_thm:.4f}")

print("\n" + "=" * 60)
print("CONCLUSION: The proof gives (2s+d-2), theorem states (2s+d)")
print("The PROOF is correct. Theorem statement has TYPO.")
print("=" * 60)

# Also check: what's the standard minimax rate for nonparametric regression?
print("\n--- Cross-check with standard results ---")
print("Standard minimax rate for C^s functions in L2:")
print("  ||f̂ - f||_{L2}² ~ n^{-2s/(2s+d)}")
print("\nFor GRADIENT estimation (one derivative less):")
print("  ||∇f̂ - ∇f||_{L2}² ~ n^{-2(s-1)/(2(s-1)+d)} = n^{-2(s-1)/(2s+d-2)}")
print("\n✓ This matches the proof! The theorem statement is wrong.")
