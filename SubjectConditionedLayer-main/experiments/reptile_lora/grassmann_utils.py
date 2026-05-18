"""
Grassmann manifold operations for Grassmann MetaLoRA.

The Grassmann manifold G(r, n) is the space of r-dimensional subspaces of R^n.
A point is represented by an orthonormal matrix Q ∈ R^(n×r) — the thin QR basis
of any matrix spanning that subspace.

Why this matters for LoRA:
    The LoRA adapter weight A ∈ R^(n×r) has a gauge symmetry:
        BA = (BC^{-1})(CA)  for any invertible C ∈ R^(r×r)
    Two matrices related this way compute identical outputs — they span the
    same subspace — yet look far apart in Euclidean space. Reptile's Euclidean
    average of adapter matrices therefore averages in the wrong space.
    Working on G(r, n) quotients out this redundancy, so the average is taken
    over the space that actually determines adapter behaviour.

Key operations
--------------
    to_grassmann(A_flat)               Project A to (Q, scale), Q orthonormal
    log_map(Q_base, Q_point)           Riemannian log: tangent vector at Q_base
    exp_map(Q_base, Delta)             Riemannian exp: move along geodesic
    grassmann_frechet_step(...)        One Reptile step on the manifold
    geodesic_distance(Q1, Q2)         ||Log_{Q1}(Q2)||_F = principal angle norm

References
----------
    Edelman, Arias, Smith (1998). The Geometry of Algorithms with Orthogonality
    Constraints. SIAM Journal on Matrix Analysis and Applications.
"""

import torch


# ── Core projection ───────────────────────────────────────────────────────────

def to_grassmann(A_flat):
    """
    Project a matrix A_flat ∈ R^(n×r) to the Grassmann manifold G(r, n).

    Uses thin QR decomposition to extract the orthonormal basis Q that
    spans the same column space as A_flat. The Frobenius norm (scale) is
    stored separately so the update can reconstruct a non-unit A.

    Args:
        A_flat (n, r): any full-column-rank matrix (the flattened adapter)

    Returns:
        Q     (n, r): orthonormal basis for colspan(A_flat)
        scale (float): Frobenius norm of A_flat — magnitude of the adapter
    """
    Q, _ = torch.linalg.qr(A_flat, mode='reduced')
    scale = A_flat.norm(p='fro').item()
    # Guard against degenerate zero initialisation
    if scale < 1e-8:
        scale = 1e-8
    return Q, scale


# ── Riemannian log map ────────────────────────────────────────────────────────

def log_map(Q_base, Q_point):
    """
    Riemannian logarithmic map on G(r, n).

    Maps Q_point into the tangent space T_{Q_base} G(r,n), returning the
    tangent vector Delta such that exp_map(Q_base, Delta) recovers Q_point.

    The construction:
        1. M = (I - Q_base Q_base^T) Q_point   — orthogonal complement component
        2. M = U diag(sigma) Vt               — thin SVD; sigma_i = sin(theta_i)
        3. Delta = U diag(arcsin(sigma)) Vt   — tangent vector with ||Delta||_F = d(Q_base, Q_point)

    where theta_i are the principal angles between the two subspaces.

    Args:
        Q_base  (n, r): orthonormal, base point on Grassmann
        Q_point (n, r): orthonormal, target point on Grassmann

    Returns:
        Delta (n, r): tangent vector at Q_base; ||Delta||_F = geodesic distance
    """
    # Component of Q_point outside the column space of Q_base
    M = Q_point - Q_base @ (Q_base.T @ Q_point)

    # Thin SVD: sigma_i = sin(theta_i), clamped away from 1 for arcsin stability
    U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clamp(max=1.0 - 1e-6)

    # Principal angles and tangent vector
    theta = sigma.arcsin()
    return U @ torch.diag(theta) @ Vt


# ── Riemannian exp map ────────────────────────────────────────────────────────

def exp_map(Q_base, Delta):
    """
    Riemannian exponential map on G(r, n).

    Moves from Q_base along the geodesic with initial velocity Delta,
    arriving at a new point on the Grassmann manifold after unit time.

    Geodesic formula (Edelman et al. 1998):
        Given Delta = U diag(theta) Vt  (thin SVD),
        Q_new = Q_base Vt^T cos(theta) + U sin(theta)

    A re-orthogonalisation step corrects floating-point drift from the
    Stiefel manifold (which represents the Grassmann manifold).

    Args:
        Q_base (n, r): orthonormal, starting point on Grassmann
        Delta  (n, r): tangent vector at Q_base

    Returns:
        Q_new (n, r): orthonormal, endpoint of geodesic
    """
    U, theta, Vt = torch.linalg.svd(Delta, full_matrices=False)

    # Move along geodesic
    Q_new = Q_base @ Vt.T @ torch.diag(theta.cos()) + U @ torch.diag(theta.sin())

    # Re-orthogonalise to correct numerical drift from the Stiefel manifold
    Q_new, _ = torch.linalg.qr(Q_new, mode='reduced')
    return Q_new


# ── Riemannian Reptile step ───────────────────────────────────────────────────

def grassmann_frechet_step(Q_base, Q_list, step_size):
    """
    One Riemannian Reptile step toward the Fréchet mean on G(r, n).

    The Fréchet mean minimises sum of squared geodesic distances to a set
    of points. Its Riemannian gradient at Q_base is:
        grad F = -2 * mean_s( Log_{Q_base}(Q_s) )

    One gradient descent step gives:
        Q_new = Exp_{Q_base}( step_size * mean_s( Log_{Q_base}(Q_s) ) )

    This is the direct Grassmann analogue of the Euclidean Reptile rule:
        theta_meta += step_size * mean_s(phi_s - theta_meta)

    Args:
        Q_base    (n, r): current meta-initialisation on Grassmann
        Q_list    list of (n, r) orthonormal matrices — subject fine-tuned subspaces
        step_size: Reptile outer step size (meta_lr)

    Returns:
        Q_new (n, r): updated orthonormal meta-initialisation
    """
    n = len(Q_list)
    # Average tangent vectors in the (locally Euclidean) tangent space
    Delta_mean = sum(log_map(Q_base, Q_s) for Q_s in Q_list) / n
    # Move along geodesic by step_size
    return exp_map(Q_base, step_size * Delta_mean)


# ── Diagnostic ────────────────────────────────────────────────────────────────

def geodesic_distance(Q1, Q2):
    """
    Geodesic distance between two points on G(r, n).

    d(Q1, Q2) = ||Log_{Q1}(Q2)||_F = sqrt( sum_i theta_i^2 )

    where theta_i are the principal angles between the two subspaces.
    Used as a health-check metric analogous to delta_norm in Euclidean Reptile.

    Args:
        Q1, Q2 (n, r): orthonormal

    Returns:
        float: geodesic distance in radians
    """
    return log_map(Q1, Q2).norm(p='fro').item()
