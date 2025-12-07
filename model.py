import time
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque

try:
    import torch
    from torchvision import models
except Exception:
    torch = None
    models = None

try:
    from domain_adaptation import DomainAdapter
except ImportError:
    DomainAdapter = None


@dataclass
class DetectorConfig:
    resize: Tuple[int, int] = (640, 360)
    window_seconds: float = 3.0
    threshold: float = 1.0
    debounce: int = 8
    cooldown_seconds: float = 1.0
    weights: Tuple[float, float, float, float] = (1.0, 0.6, 0.6, 1.0)
    use_cnn: bool = True
    cnn_weight: float = 1.0
    # Domain adaptation parameters
    use_domain_adaptation: bool = True
    feature_buffer_size: int = 30
    mmd_weight: float = 0.8
    mmd_sigma: float = 1.0
    # Supervised domain adaptation with labeled data
    use_labeled_domains: bool = False
    labeled_domains_path: Optional[str] = None
    coral_weight: float = 0.5


class DomainShiftDetector:

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.cfg = config
        self.ema_gray: Optional[np.ndarray] = None
        self.ema_hist: Optional[np.ndarray] = None
        self.ema_brightness: Optional[float] = None
        self.ema_edge: Optional[float] = None
        self.ema_embed: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None
        self.exceed_count: int = 0
        self.last_event_time: float = -1e9

        self.last_score: float = 0.0
        self.last_components: Dict[str, float] = {}
        self.last_label: str = ""

        self.cnn = self._init_cnn() if self.cfg.use_cnn else None
        
        # Domain adaptation components
        if self.cfg.use_domain_adaptation and self.cfg.use_cnn:
            self.feature_buffer: deque = deque(maxlen=self.cfg.feature_buffer_size)
            self.baseline_mean: Optional[np.ndarray] = None
            self.baseline_cov: Optional[np.ndarray] = None
            
            # Supervised domain adaptation with labeled source domains
            if self.cfg.use_labeled_domains and DomainAdapter is not None:
                self.domain_adapter = DomainAdapter()
                if self.cfg.labeled_domains_path:
                    try:
                        self.domain_adapter.load_source_domains(self.cfg.labeled_domains_path)
                    except Exception:
                        pass
            else:
                self.domain_adapter = None
        else:
            self.feature_buffer = None
            self.baseline_mean = None
            self.baseline_cov = None
            self.domain_adapter = None

    def _alpha(self, now: float) -> float:
        if self.last_time is None:
            return 1.0
        dt = max(1e-6, now - self.last_time)
        return 1.0 - math.exp(-dt / max(1e-6, self.cfg.window_seconds))

    def _prep(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = self.cfg.resize[1], self.cfg.resize[0]
        frame_r_bgr = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_r_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_r_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return frame_r_bgr, hsv, gray, edges

    def _init_cnn(self):
        if torch is None or models is None:
            return None
        try:
            try:
                base = models.mobilenet_v2(pretrained=True)
            except Exception:
                base = models.mobilenet_v2(weights=getattr(models, "MobileNet_V2_Weights", None).DEFAULT)
        except Exception:
            return None
        backbone = torch.nn.Sequential(
            base.features,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        return backbone

    def _cnn_embed(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.cnn is None or torch is None:
            return None
        try:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            t = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                feat = self.cnn(t).cpu().numpy().reshape(-1)
            n = np.linalg.norm(feat) + 1e-8
            return (feat / n).astype(np.float32)
        except Exception:
            return None

    def _hist_hs(self, hsv: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        return hist.flatten()

    def _brightness(self, hsv: np.ndarray) -> float:
        v = hsv[:, :, 2].astype(np.float32)
        return float(v.mean() / 255.0)

    def _edge_density(self, edges: np.ndarray) -> float:
        return float((edges > 0).mean())

    def _ssim_to_ema(self, gray: np.ndarray) -> float:
        if self.ema_gray is None:
            return 0.0
        try:
            score = ssim(gray, self.ema_gray.astype(np.uint8), data_range=255)
        except Exception:
            score = 1.0
        return float(1.0 - score)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        # Compute pairwise squared distances
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
        dists = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        # Apply RBF kernel
        return np.exp(-dists / (2 * sigma ** 2))

    def _mmd_distance(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        if X.size == 0 or Y.size == 0:
            return 0.0
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        if X.shape[1] != Y.shape[1]:
            return 0.0
        
        try:
            # Compute kernel matrices
            K_XX = self._rbf_kernel(X, X, sigma)
            K_YY = self._rbf_kernel(Y, Y, sigma)
            K_XY = self._rbf_kernel(X, Y, sigma)
            
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            mmd_squared = (
                np.mean(K_XX) +
                np.mean(K_YY) -
                2 * np.mean(K_XY)
            )
            
            return float(max(0.0, np.sqrt(mmd_squared)))
        except Exception:
            return 0.0

    def _update_distribution_stats(self, features: List[np.ndarray]) -> None:
        if not features or len(features) < 2:
            return
        
        try:
            # Stack features into matrix
            feature_matrix = np.vstack(features)
            
            # Compute mean and covariance
            self.baseline_mean = np.mean(feature_matrix, axis=0).astype(np.float32)
            centered = feature_matrix - self.baseline_mean
            self.baseline_cov = np.cov(centered.T).astype(np.float32)
            
            # Regularize covariance to avoid singularity
            reg = 1e-6 * np.eye(self.baseline_cov.shape[0], dtype=np.float32)
            self.baseline_cov += reg
        except Exception:
            self.baseline_mean = None
            self.baseline_cov = None

    def _mahalanobis_distance(self, feature: np.ndarray) -> float:
        if self.baseline_mean is None or self.baseline_cov is None:
            return 0.0
        
        try:
            diff = feature - self.baseline_mean
            inv_cov = np.linalg.pinv(self.baseline_cov)
            dist_sq = np.dot(np.dot(diff, inv_cov), diff)
            return float(np.sqrt(max(0.0, dist_sq)))
        except Exception:
            return 0.0

    def step(self, frame: np.ndarray, now: Optional[float] = None) -> Optional[Dict]:

        if now is None:
            now = time.time()

        frame_r_bgr, hsv, gray, edges = self._prep(frame)
        hist = self._hist_hs(hsv)
        brightness = self._brightness(hsv)
        edge = self._edge_density(edges)
        ssim_dist = self._ssim_to_ema(gray)
        embed = self._cnn_embed(frame_r_bgr) if self.cfg.use_cnn else None

        alpha = self._alpha(now)
        if self.ema_hist is None:
            self.ema_hist = hist.copy()
            self.ema_brightness = brightness
            self.ema_edge = edge
            self.ema_gray = gray.astype(np.uint8)
            if embed is not None:
                self.ema_embed = embed.copy()
                # Initialize domain adaptation components
                if self.feature_buffer is not None:
                    self.feature_buffer.append(embed.copy())
            self.last_time = now
            self.last_score = 0.0
            self.last_components = {"hist": 0.0, "brightness": 0.0, "edge": 0.0, "ssim": 0.0, "cnn": 0.0}
            self.last_label = ""
            return None

        hist_l1 = float(np.abs(hist - self.ema_hist).sum())
        bright_d = float(abs(brightness - (self.ema_brightness or 0.0)))
        edge_d = float(abs(edge - (self.ema_edge or 0.0)))

        w_hist, w_b, w_e, w_s = self.cfg.weights
        score = w_hist * hist_l1 + w_b * bright_d + w_e * edge_d + w_s * ssim_dist
        cnn_d = 0.0
        mmd_d = 0.0
        maha_d = 0.0
        coral_d = 0.0
        source_mmd_d = 0.0
        
        if embed is not None and self.ema_embed is not None:
            cnn_d = float(1.0 - float(np.dot(embed, self.ema_embed)))
            score += self.cfg.cnn_weight * cnn_d
            
            # Domain adaptation features
            if self.cfg.use_domain_adaptation and self.feature_buffer is not None:
                # Add current embedding to feature buffer
                self.feature_buffer.append(embed.copy())
                
                # Update distribution statistics periodically
                if len(self.feature_buffer) >= 10:
                    self._update_distribution_stats(list(self.feature_buffer))
                
                # Unsupervised: Compute MMD distance between current embedding and online baseline
                if len(self.feature_buffer) >= 5:
                    baseline_features = np.array(list(self.feature_buffer)[:-1])
                    current_batch = embed.reshape(1, -1)
                    mmd_d = self._mmd_distance(current_batch, baseline_features, sigma=self.cfg.mmd_sigma)
                    
                    # Normalize MMD to [0, 1] range for scoring
                    mmd_d_normalized = min(1.0, mmd_d / 2.0)  # Heuristic normalization
                    score += self.cfg.mmd_weight * mmd_d_normalized
                
                # Supervised: Compare against labeled source domains (if available)
                if self.domain_adapter is not None and self.domain_adapter.source_domains:
                    # Find closest source domain
                    closest_source, source_dist = self.domain_adapter.find_closest_source_domain(embed)
                    
                    # Compute MMD to closest source domain (supervised comparison)
                    if closest_source:
                        source_mmd_d = self.domain_adapter.compute_mmd_to_source(
                            embed, closest_source, sigma=self.cfg.mmd_sigma
                        )
                        source_mmd_normalized = min(1.0, source_mmd_d / 2.0)
                        score += self.cfg.mmd_weight * 0.6 * source_mmd_normalized
                    
                    # Compute CORAL alignment distance (supervised domain adaptation technique)
                    if len(self.feature_buffer) >= 10 and closest_source:
                        recent_features = np.array(list(self.feature_buffer)[-10:])
                        coral_d = self.domain_adapter.compute_coral_alignment(closest_source, recent_features)
                        coral_normalized = min(1.0, coral_d / 100.0)  # Normalize CORAL distance
                        score += self.cfg.coral_weight * coral_normalized
                
                # Compute Mahalanobis distance from baseline distribution
                if self.baseline_mean is not None and self.baseline_cov is not None:
                    maha_d = self._mahalanobis_distance(embed)
                    # Normalize Mahalanobis distance (typically ranges from 0 to ~5-10)
                    maha_d_normalized = min(1.0, maha_d / 5.0)
                    score += self.cfg.mmd_weight * 0.5 * maha_d_normalized

        self.ema_hist = (1.0 - alpha) * self.ema_hist + alpha * hist
        self.ema_brightness = (1.0 - alpha) * (self.ema_brightness or brightness) + alpha * brightness
        self.ema_edge = (1.0 - alpha) * (self.ema_edge or edge) + alpha * edge
        self.ema_gray = cv2.addWeighted(self.ema_gray.astype(np.float32), (1.0 - alpha),
                                        gray.astype(np.float32), alpha, 0.0).clip(0, 255).astype(np.uint8)
        if embed is not None:
            if self.ema_embed is None:
                self.ema_embed = embed.copy()
            else:
                self.ema_embed = (1.0 - alpha) * self.ema_embed + alpha * embed
                n = np.linalg.norm(self.ema_embed) + 1e-8
                self.ema_embed = (self.ema_embed / n).astype(np.float32)

        self.last_time = now
        self.last_components = {
            "hist": hist_l1,
            "brightness": bright_d,
            "edge": edge_d,
            "ssim": ssim_dist,
            "cnn": cnn_d,
        }
        
        # Add domain adaptation components if enabled
        if self.cfg.use_domain_adaptation:
            self.last_components["mmd"] = mmd_d
            self.last_components["mahalanobis"] = maha_d
            if self.cfg.use_labeled_domains:
                self.last_components["source_mmd"] = source_mmd_d
                self.last_components["coral"] = coral_d
        
        self.last_score = score

        if score > self.cfg.threshold:
            self.exceed_count += 1
        else:
            self.exceed_count = 0

        if (
            self.exceed_count >= self.cfg.debounce
            and (now - self.last_event_time) >= self.cfg.cooldown_seconds
        ):
            label = self._categorize(hist_l1, bright_d, edge_d, ssim_dist)
            self.last_label = label
            self.last_event_time = now
            self.exceed_count = 0
            return {
                "time": now,
                "score": score,
                "label": label,
                "components": self.last_components.copy(),
                "threshold": self.cfg.threshold,
            }

        return None

    def _categorize(self, hist_l1: float, bright_d: float, edge_d: float, ssim_d: float) -> str:
        if bright_d >= 0.15 and hist_l1 < 0.30 and edge_d < 0.10:
            return "lighting_change"
        if ssim_d >= 0.50 and edge_d >= 0.10 and bright_d < 0.20:
            return "camera_motion"
        return "scene_or_object_change"


__all__ = ["DetectorConfig", "DomainShiftDetector"]

