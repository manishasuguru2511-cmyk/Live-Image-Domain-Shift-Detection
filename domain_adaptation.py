import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SourceDomain:
    name: str
    features: np.ndarray  # N x D matrix of feature vectors
    mean: np.ndarray  # Mean feature vector
    cov: np.ndarray  # Covariance matrix
    domain_type: str  # e.g., "indoor", "outdoor", "daytime", "nighttime"


class DomainAdapter:
    
    def __init__(self):
        self.source_domains: Dict[str, SourceDomain] = {}
        
    def add_labeled_source_domain(self, name: str, features: List[np.ndarray], domain_type: str = "unknown") -> None:
        if not features:
            return
        
        feature_matrix = np.vstack(features)
        mean = np.mean(feature_matrix, axis=0).astype(np.float32)
        centered = feature_matrix - mean
        cov = np.cov(centered.T).astype(np.float32)
        
        # Regularize covariance
        reg = 1e-6 * np.eye(cov.shape[0], dtype=np.float32)
        cov += reg
        
        self.source_domains[name] = SourceDomain(
            name=name,
            features=feature_matrix,
            mean=mean,
            cov=cov,
            domain_type=domain_type
        )
    
    def compute_mmd_to_source(self, current_features: np.ndarray, source_name: str, sigma: float = 1.0) -> float:
        if source_name not in self.source_domains:
            return 0.0
        
        source = self.source_domains[source_name]
        
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)
        
        try:
            # RBF kernel matrices
            K_XX = self._rbf_kernel(source.features, source.features, sigma)
            K_YY = self._rbf_kernel(current_features, current_features, sigma)
            K_XY = self._rbf_kernel(source.features, current_features, sigma)
            
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            mmd_squared = (
                np.mean(K_XX) +
                np.mean(K_YY) -
                2 * np.mean(K_XY)
            )
            
            return float(max(0.0, np.sqrt(mmd_squared)))
        except Exception:
            return 0.0
    
    def compute_coral_alignment(self, source_name: str, target_features: np.ndarray) -> float:
        if source_name not in self.source_domains:
            return 0.0
        
        source = self.source_domains[source_name]
        
        if target_features.ndim == 1:
            target_features = target_features.reshape(1, -1)
        
        try:
            # Compute target covariance
            target_mean = np.mean(target_features, axis=0)
            target_centered = target_features - target_mean
            target_cov = np.cov(target_centered.T)
            
            # CORAL distance: Frobenius norm of covariance difference
            cov_diff = source.cov - target_cov
            coral_dist = np.linalg.norm(cov_diff, 'fro')
            
            return float(coral_dist)
        except Exception:
            return 0.0
    
    def find_closest_source_domain(self, current_feature: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.source_domains:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_name = None
        
        for name, domain in self.source_domains.items():
            # Use Mahalanobis distance to source domain
            try:
                diff = current_feature - domain.mean
                inv_cov = np.linalg.pinv(domain.cov)
                dist_sq = np.dot(np.dot(diff, inv_cov), diff)
                dist = np.sqrt(max(0.0, dist_sq))
                
                if dist < min_dist:
                    min_dist = dist
                    closest_name = name
            except Exception:
                continue
        
        return closest_name, min_dist
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
        dists = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        return np.exp(-dists / (2 * sigma ** 2))
    
    def save_source_domains(self, filepath: str) -> None:
        data = {}
        for name, domain in self.source_domains.items():
            data[name] = {
                "domain_type": domain.domain_type,
                "mean": domain.mean.tolist(),
                "cov": domain.cov.tolist(),
                "num_examples": len(domain.features),
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_source_domains(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.source_domains = {}
        for name, info in data.items():
            mean = np.array(info["mean"], dtype=np.float32)
            cov = np.array(info["cov"], dtype=np.float32)
            features = np.array([mean])
            
            self.source_domains[name] = SourceDomain(
                name=name,
                features=features,
                mean=mean,
                cov=cov,
                domain_type=info.get("domain_type", "unknown")
            )


__all__ = ["DomainAdapter", "SourceDomain"]

