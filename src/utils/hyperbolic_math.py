import torch
import numpy as np

def poincare_distance(u, v, eps=1e-5):
    """
    Compute Poincaré distance between points u and v in the Poincaré ball.
    
    Distance formula: d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    
    Args:
        u: torch.Tensor of shape (..., dim)
        v: torch.Tensor of shape (..., dim)
        eps: small constant for numerical stability
    
    Returns:
        torch.Tensor of distances
    """
    sqrt_u = torch.sum(u ** 2, dim=-1, keepdim=True)
    sqrt_v = torch.sum(v ** 2, dim=-1, keepdim=True)
    
    sqrt_u = torch.clamp(sqrt_u, 0, 1 - eps)
    sqrt_v = torch.clamp(sqrt_v, 0, 1 - eps)
    
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1, keepdim=True)
    
    numerator = 2 * diff_norm_sq
    denominator = (1 - sqrt_u) * (1 - sqrt_v)
    
    delta = 1 + numerator / (denominator + eps)
    delta = torch.clamp(delta, min=1.0 + eps)
    
    distance = torch.acosh(delta)
    
    return distance.squeeze(-1)

def poincare_distance_numpy(u, v, eps=1e-5):
    """
    Numpy version of Poincaré distance.
    
    Args:
        u: numpy array of shape (..., dim)
        v: numpy array of shape (..., dim)
        eps: small constant for numerical stability
    
    Returns:
        numpy array of distances
    """
    sqrt_u = np.sum(u ** 2, axis=-1, keepdims=True)
    sqrt_v = np.sum(v ** 2, axis=-1, keepdims=True)
    
    sqrt_u = np.clip(sqrt_u, 0, 1 - eps)
    sqrt_v = np.clip(sqrt_v, 0, 1 - eps)
    
    diff_norm_sq = np.sum((u - v) ** 2, axis=-1, keepdims=True)
    
    numerator = 2 * diff_norm_sq
    denominator = (1 - sqrt_u) * (1 - sqrt_v)
    
    delta = 1 + numerator / (denominator + eps)
    delta = np.clip(delta, a_min=1.0 + eps, a_max=None)
    
    distance = np.arccosh(delta)
    
    return distance.squeeze(-1)

def project_to_poincare_ball(x, eps=1e-5):
    """
    Project points to the Poincaré ball (norm < 1).
    
    Args:
        x: torch.Tensor of shape (..., dim)
        eps: small constant for numerical stability
    
    Returns:
        torch.Tensor projected to Poincaré ball
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = 1 - eps
    
    scale = torch.where(norm > max_norm, max_norm / (norm + eps), torch.ones_like(norm))
    
    return x * scale

def exponential_map(x, v, c=1.0):
    """
    Exponential map at point x with tangent vector v.
    Maps from tangent space to the manifold.
    
    Args:
        x: point on manifold, shape (..., dim)
        v: tangent vector at x, shape (..., dim)
        c: curvature (default 1.0 for unit ball)
    
    Returns:
        point on manifold
    """
    sqrt_c = c ** 0.5
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-10)
    second_term = (
        torch.tanh(sqrt_c * lambda_x(x, c) * v_norm / 2)
        * v / (sqrt_c * v_norm)
    )
    return mobius_add(x, second_term, c)

def lambda_x(x, c=1.0):
    """
    Conformal factor at point x.
    
    Args:
        x: point on manifold
        c: curvature
    
    Returns:
        conformal factor
    """
    return 2 / (1 - c * torch.sum(x ** 2, dim=-1, keepdim=True)).clamp(min=1e-10)

def mobius_add(x, y, c=1.0):
    """
    Möbius addition in the Poincaré ball.
    
    Args:
        x, y: points in Poincaré ball
        c: curvature
    
    Returns:
        x ⊕ y in Poincaré ball
    """
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c ** 2 * x2 * y2
    
    return numerator / denominator.clamp(min=1e-10)

def riemannian_gradient(euclidean_grad, x, c=1.0):
    """
    Convert Euclidean gradient to Riemannian gradient.
    
    Args:
        euclidean_grad: Euclidean gradient
        x: point on manifold
        c: curvature
    
    Returns:
        Riemannian gradient
    """
    scale = ((1 - c * torch.sum(x ** 2, dim=-1, keepdim=True)) ** 2) / 4
    return scale * euclidean_grad
