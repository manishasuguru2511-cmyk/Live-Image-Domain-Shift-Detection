# How Labeled Data is Used for Domain Adaptation

## Your Question: "How exactly? They were talking about usage of labeled data"

Great question! Let me explain the difference between what we had (unsupervised) and what domain adaptation research discusses (supervised with labeled data).

## Current Implementation (Before): Unsupervised

**What we had:**
- **Feature Buffer**: Stores recent CNN embeddings from the video stream itself
- **Baseline Statistics**: Computes mean and covariance from the buffer (online, adaptive)
- **Distribution Comparison**: Compares current frame to this online baseline

**Problem**: This builds a baseline from the stream itself - it's **adaptive but doesn't use labeled training data**.

## What the Article Discusses: Supervised Domain Adaptation

**The approach uses labeled source domain examples:**

1. **Labeled Source Domains**: Pre-trained on labeled examples of known scene types
   - Example: "normal_indoor_scene", "outdoor_daytime", "indoor_nighttime"
   - These are labeled training examples you collect beforehand

2. **Domain-Invariant Feature Learning**: 
   - Train on labeled examples to learn what "normal" scenes look like
   - Learn features that distinguish between different scene types

3. **Comparison Against Labeled Domains**:
   - Compare current stream against these **pre-labeled source domains**
   - Detect shifts when current stream diverges from known domains

## How Labeled Data is Used (Implementation)

### Step 1: Training Phase (Using Labeled Data)

```python
# Collect labeled example videos of different scene types
# "normal_indoor.mp4" - labeled as "indoor" domain
# "normal_outdoor.mp4" - labeled as "outdoor" domain

# Extract features from labeled videos
features_indoor = extract_features("normal_indoor.mp4")
features_outdoor = extract_features("normal_outdoor.mp4")

# Create labeled source domains
domain_adapter.add_labeled_source_domain(
    name="indoor_daytime",
    features=features_indoor,  # Labeled examples!
    domain_type="indoor"
)

domain_adapter.add_labeled_source_domain(
    name="outdoor_daytime", 
    features=features_outdoor,  # Labeled examples!
    domain_type="outdoor"
)
```

### Step 2: Detection Phase (Using Trained Domains)

When processing a video stream, the system:

1. **Extracts features** from current frame (CNN embedding)
2. **Compares against labeled source domains**:
   - Computes MMD distance to each labeled source domain
   - Finds closest labeled source domain
   - Measures how far current stream is from known domains
3. **Detects domain shift** when current stream is far from all labeled source domains

## Key Techniques Using Labeled Data

### 1. Maximum Mean Discrepancy (MMD) to Source Domains
```
Current Feature → Compare with Labeled Source Domain Features → MMD Distance
```
- Uses **labeled source domain examples** (not just online baseline)
- Measures distribution divergence using kernel mean embeddings
- This is the **supervised** approach - comparing to pre-labeled domains

### 2. CORAL (Correlation Alignment)
```
Current Stream Covariance → Compare with Labeled Source Domain Covariance → CORAL Distance
```
- Aligns second-order statistics between current stream and labeled source domains
- Key technique in supervised domain adaptation
- Uses labeled training data to learn domain-invariant features

### 3. Feature Buffer (Hybrid Approach)
We use **both**:
- **Online baseline** (unsupervised) - adapts to current stream
- **Labeled source domains** (supervised) - compares to known scene types

## Workflow Example

### 1. Collect Labeled Training Data
```
You have videos of:
- "normal_indoor.mp4" - labeled as "indoor" domain
- "normal_outdoor.mp4" - labeled as "outdoor" domain  
- "normal_daytime.mp4" - labeled as "daytime" domain
```

### 2. Train Source Domains (Using Labeled Data)
```bash
python train_source_domains.py \
    --video normal_indoor.mp4 \
    --domain-name "indoor" \
    --domain-type "indoor" \
    --output domains.json

python train_source_domains.py \
    --video normal_outdoor.mp4 \
    --domain-name "outdoor" \
    --domain-type "outdoor" \
    --output domains.json \
    --append
```

### 3. Use in Detection
```python
config = DetectorConfig(
    use_domain_adaptation=True,
    use_labeled_domains=True,  # Enable supervised approach
    labeled_domains_path="domains.json",  # Use labeled training data!
)
```

## The Difference

| Aspect | Unsupervised (Before) | Supervised (With Labeled Data) |
|--------|----------------------|-------------------------------|
| **Training Data** | None - builds baseline from stream | Uses labeled examples from known scene types |
| **Baseline** | Online, adaptive from current video | Pre-trained from labeled source domains |
| **Comparison** | Current frame vs. online baseline | Current frame vs. labeled source domains |
| **Domain Awareness** | Generic change detection | Knows specific domain types (indoor/outdoor) |
| **Robustness** | Sensitive to gradual drift | More robust - compares to fixed labeled references |

## Why This Matters

The article emphasizes that using **labeled source domain examples**:
- Provides better baselines (trained on known good examples)
- Enables domain-aware detection (knows what "normal" looks like)
- More robust to false positives (compares to real labeled examples)
- Aligns with domain adaptation research (transfer learning approach)

## Summary

**Your question**: "How exactly? They were talking about usage of labeled data"

**Answer**: The labeled data approach means:
1. **Training phase**: Extract features from labeled example videos (e.g., "normal_indoor", "normal_outdoor")
2. **Create source domains**: Build statistical models (mean, covariance) from labeled examples
3. **Detection phase**: Compare current stream features against these labeled source domain models
4. **Domain shift**: Detected when current stream diverges from all labeled source domains

This is the **supervised domain adaptation** approach the article discusses - using labeled training data to learn domain-invariant features and build better baselines for domain shift detection!

