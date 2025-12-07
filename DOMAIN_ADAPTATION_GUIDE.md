# Supervised Domain Adaptation Guide

This guide explains how to use **labeled data for domain adaptation**, as discussed in domain adaptation research articles.

## Understanding the Approach

### Current Implementation (Unsupervised)
- **Feature Buffer**: Stores recent CNN embeddings from the video stream
- **Baseline Statistics**: Computes mean and covariance from the buffer
- **Distribution Comparison**: Compares current frame to this online baseline

This works well but doesn't leverage **labeled training data**.

### Supervised Domain Adaptation (Using Labeled Data)

As discussed in domain adaptation research, you can use **labeled source domain examples** to:

1. **Learn Domain-Invariant Features**: Train on labeled examples from known scene types
2. **Build Source Domain Baselines**: Create statistical models of known domains (e.g., "indoor", "outdoor", "daytime", "nighttime")
3. **Compare Against Labeled Domains**: Detect shifts by comparing current stream to these labeled source domains

## How It Works

### 1. Training Labeled Source Domains

Create labeled source domains from example videos:

```bash
# Train a source domain from a video showing "normal" indoor scenes
python train_source_domains.py \
    --video "examples/indoor_daytime.mp4" \
    --domain-name "indoor_daytime" \
    --domain-type "indoor" \
    --output "source_domains.json"

# Train another source domain for outdoor scenes
python train_source_domains.py \
    --video "examples/outdoor_daytime.mp4" \
    --domain-name "outdoor_daytime" \
    --domain-type "outdoor" \
    --output "source_domains.json" \
    --append
```

### 2. Using Labeled Domains in Detection

The detector now compares the current stream against these labeled source domains:

```python
from model import DetectorConfig, DomainShiftDetector

config = DetectorConfig(
    use_domain_adaptation=True,
    use_labeled_domains=True,  # Enable supervised approach
    labeled_domains_path="source_domains.json",  # Path to trained domains
    mmd_weight=0.8,
    coral_weight=0.5,
)

detector = DomainShiftDetector(config)
```

### 3. Domain Adaptation Techniques Used

#### Maximum Mean Discrepancy (MMD)
- Compares feature distributions between current stream and labeled source domains
- Uses kernel mean embeddings (RBF kernel)
- Measures how far the current domain is from known source domains

#### CORAL (Correlation Alignment)
- Aligns second-order statistics (covariance) between source and target
- Compares the covariance structure of features
- Key technique for supervised domain adaptation

#### Mahalanobis Distance
- Measures statistical distance from labeled source domain distributions
- Accounts for feature correlations
- More robust than Euclidean distance

## Workflow Example

### Step 1: Collect Training Videos
- Record videos of different scene types you want to detect
- Examples: "normal_indoor.mp4", "normal_outdoor.mp4", "daytime_scene.mp4"

### Step 2: Train Source Domains
```bash
python train_source_domains.py --video normal_indoor.mp4 --domain-name "normal_indoor" --output domains.json
python train_source_domains.py --video normal_outdoor.mp4 --domain-name "normal_outdoor" --output domains.json --append
```

### Step 3: Use in Detection
When running detection, the system will:
- Compare current stream features against labeled source domains
- Compute MMD distance to each source domain
- Compute CORAL alignment distance
- Detect domain shifts when current stream diverges from all known source domains

## Benefits of Using Labeled Data

1. **Better Baseline**: Uses pre-labeled "normal" scenes as reference instead of online baseline
2. **Domain-Aware Detection**: Can identify specific domain types (indoor vs outdoor, day vs night)
3. **More Robust**: Less sensitive to gradual drift, better at detecting true domain shifts
4. **Interpretable**: Can report which known domain is closest to current stream

## Technical Details

### Feature Buffer (Unsupervised)
- Stores last N CNN embeddings from stream
- Builds baseline from current video itself
- Adapts online as video plays

### Labeled Source Domains (Supervised)
- Pre-computed from labeled training videos
- Represents known scene types/domains
- Fixed reference points for comparison

### Combined Approach
- Uses both: compares against online baseline AND labeled source domains
- More robust detection by leveraging both approaches
- Domain shift detected when current stream diverges from all known domains

## Integration with Current System

The supervised approach integrates seamlessly:
- If `use_labeled_domains=False`: Uses only unsupervised approach (current behavior)
- If `use_labeled_domains=True`: Uses both unsupervised AND supervised approaches
- Components are added to event outputs: `source_mmd`, `coral` metrics appear in JSON/CSV

This makes the system production-ready with both approaches available!

