import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List
import time

try:
    import torch
    from torchvision import models
except Exception:
    torch = None
    models = None

from domain_adaptation import DomainAdapter


def extract_features_from_video(video_path: str, max_frames: int = 100) -> List[np.ndarray]:
    if torch is None or models is None:
        raise RuntimeError("PyTorch and torchvision are required for feature extraction")
    
    # Initialize CNN
    try:
        base = models.mobilenet_v2(pretrained=True)
    except Exception:
        base = models.mobilenet_v2(weights=getattr(models, "MobileNet_V2_Weights", None).DEFAULT)
    
    backbone = torch.nn.Sequential(
        base.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / 2))  # Sample every 0.5 seconds
    
    features = []
    frame_count = 0
    processed = 0
    
    print(f"Extracting features from {video_path}...")
    
    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Extract feature
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            t = torch.from_numpy(x).unsqueeze(0)
            
            with torch.no_grad():
                feat = backbone(t).cpu().numpy().reshape(-1)
            
            # L2 normalize
            n = np.linalg.norm(feat) + 1e-8
            feat = (feat / n).astype(np.float32)
            features.append(feat)
            processed += 1
            
            if processed % 10 == 0:
                print(f"  Processed {processed}/{max_frames} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(features)} feature vectors")
    return features


def main():
    parser = argparse.ArgumentParser(description="Train labeled source domains from videos")
    parser.add_argument("--video", type=str, required=True, help="Path to video file for training")
    parser.add_argument("--domain-name", type=str, required=True, help="Name for this source domain (e.g., 'indoor_daytime')")
    parser.add_argument("--domain-type", type=str, default="unknown", help="Type of domain (e.g., 'indoor', 'outdoor')")
    parser.add_argument("--output", type=str, default="source_domains.json", help="Output file for domain statistics")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to extract (default: 100)")
    parser.add_argument("--append", action="store_true", help="Append to existing domains file")
    
    args = parser.parse_args()
    
    # Extract features from video
    features = extract_features_from_video(args.video, args.max_frames)
    
    if not features:
        print("Error: No features extracted from video")
        return 1
    
    # Create or load domain adapter
    adapter = DomainAdapter()
    output_path = Path(args.output)
    
    if args.append and output_path.exists():
        try:
            adapter.load_source_domains(str(output_path))
            print(f"Loaded existing domains from {output_path}")
        except Exception as e:
            print(f"Warning: Could not load existing domains: {e}")
    
    # Add new labeled source domain
    adapter.add_labeled_source_domain(
        name=args.domain_name,
        features=features,
        domain_type=args.domain_type
    )
    
    # Save
    adapter.save_source_domains(str(output_path))
    print(f"\nSaved source domain '{args.domain_name}' with {len(features)} examples to {output_path}")
    print(f"Domain type: {args.domain_type}")
    print(f"Total domains: {len(adapter.source_domains)}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

