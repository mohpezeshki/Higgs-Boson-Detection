import numpy as np
import random
import torch
import logging

logging.basicConfig(level=logging.INFO)

def statistical_summary(dataframe):
    print("="*60)
    print("EDA SUMMARY")
    print("="*60)

    # Extract data from dataframe
    y = dataframe['label'].values
    X = dataframe.iloc[:, 1:].values

    # 1. CLASS DISTRIBUTION
    background_count = (y == 0).sum()
    signal_count = (y == 1).sum()
    background_pct = (y == 0).mean() * 100
    signal_pct = (y == 1).mean() * 100
    imbalance_ratio = max(background_count, signal_count) / min(background_count, signal_count)

    print(f"\n1. CLASS DISTRIBUTION:")
    print(f"   Background (0): {background_count:,} ({background_pct:.1f}%)")
    print(f"   Signal (1): {signal_count:,} ({signal_pct:.1f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 10:
        print(f"   SEVERE IMBALANCE: Must use class weights")
        use_class_weights = True
    elif imbalance_ratio > 1.5:
        print(f"   MODERATE IMBALANCE: Class weights recommended")
        use_class_weights = True
    else:
        print(f"   BALANCED DATA: Class weights optional")
        use_class_weights = False

    # 2. MISSING VALUES
    missing_count = dataframe.isnull().sum().sum()
    print(f"\n2. MISSING VALUES:")
    print(f"   Total missing: {missing_count}")
    if missing_count == 0:
        print(f" ✓ OK (data quality is excellent)")
        has_missing = False
    else:
        print(f" ✗ Need to handle missing values")
        has_missing = True

    # 3. FEATURE VARIANCE
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    variance = np.var(X_scaled, axis=0)

    low_var_threshold = 0.01
    low_var_count = (variance < low_var_threshold).sum()
    low_var_features = np.where(variance < low_var_threshold)[0]

    print(f"\n3. FEATURE VARIANCE:")
    print(f"   Min variance: {variance.min():.6f}")
    print(f"   Max variance: {variance.max():.6f}")
    print(f"   Mean variance: {variance.mean():.6f}")
    print(f"   Features with variance < {low_var_threshold}: {low_var_count}")

    if low_var_count > 0:
        print(f"   Low-variance features: {low_var_features.tolist()}")
        print(f" ✗ Remove {low_var_count} features")
        remove_low_variance = True
    else:
        print(f"   ✓ OK (all features have good variance)")
        remove_low_variance = False

    # 4. OUTLIERS (check max/min values)
    print(f"\n4. OUTLIERS:")
    max_values = X_scaled.max(axis=0)
    min_values = X_scaled.min(axis=0)
    extreme_outliers = np.where((np.abs(max_values) > 10) | (np.abs(min_values) > 10))[0]

    print(f"   Max value across features: {max_values.max():.2f}")
    print(f"   Min value across features: {min_values.min():.2f}")
    print(f"   Features with extreme outliers (|x| > 10): {len(extreme_outliers)}")

    if len(extreme_outliers) > 0:
        print(f"   Extreme feature indices: {extreme_outliers.tolist()}")
        print(f"   Use gradient clipping during training")
        use_gradient_clipping = True
    else:
        print(f"   ✓ OK (no extreme outliers)")
        use_gradient_clipping = False

    # 5. CLASS SEPARABILITY (check distribution differences)
    print(f"\n5. CLASS SEPARABILITY:")
    mean_background = X[y == 0].mean(axis=0)
    mean_signal = X[y == 1].mean(axis=0)
    separation = np.abs(mean_background - mean_signal).mean()

    print(f"   Mean separation between classes: {separation:.4f}")
    if separation > 0.1:
        print(f"   ✓ Classes are well separated (problem is solvable)")
        separable = True
    elif separation > 0.05:
        print(f"   Moderate separation (problem is solvable)")
        separable = True
    else:
        print(f"   Weak separation (problem may be hard)")
        separable = True  # Still solvable with deep learning

    # 6. PREPROCESSING NEEDED
    print(f"\n6. PREPROCESSING STRATEGY:")
    print(f"   ✓ Standardize features (StandardScaler)")
    print(f"   ✓ Stratified train/val/test split (preserve class distribution)")

    if use_class_weights:
        print(f"   ✓ Use class weights (handle imbalance)")
    else:
        print(f"   ○ Class weights optional (data is balanced)")

    if use_gradient_clipping:
        print(f"   ✓ Use gradient clipping (clip_norm=1.0)")

    if remove_low_variance:
        print(f"   ✓ Remove {low_var_count} low-variance features")

    # 7. SUMMARY RECOMMENDATIONS
    print(f"\n7. TRAINING RECOMMENDATIONS:")
    print(f"   Model architecture: 5-layer feedforward NN")
    print(f"   Batch size: 1024")
    print(f"   Learning rate: 0.001")
    print(f"   Optimizer: Adam with L2 regularization (weight_decay=1e-6)")
    print(f"   Dropout rate: 0.2")

    if use_class_weights:
        print(f"   Loss function: CrossEntropyLoss(weight=class_weights)")
    else:
        print(f"   Loss function: CrossEntropyLoss (no weights)")

    if use_gradient_clipping:
        print(f"   Gradient clipping: clip_norm=1.0")

    print(f"   Learning rate schedule: ReduceLROnPlateau")

    # 8. DECISION VARIABLES (for later use in training script)
    print(f"\n" + "="*60)
    print("DECISION VARIABLES FOR TRAINING")
    print("="*60)
    print(f"use_class_weights: {use_class_weights}")
    print(f"use_gradient_clipping: {use_gradient_clipping}")
    print(f"remove_low_variance: {remove_low_variance}")
    print(f"low_variance_features: {low_var_features.tolist()}")
    print("="*60)


def print_basic_info(dataframe):
    print("First 5 rows of the dataset:")
    print(dataframe.head())
    print(f"Dataset shape: {dataframe.shape}")
    print(f"Dataset length: {len(dataframe)}")
    print(f"Dataset data types:\n{dataframe.dtypes}")
    print(f"Description of the dataset:\n{dataframe.describe()}")

def set_random_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def get_class_weights(y: np.ndarray) -> torch.FloatTensor:
    """Compute class weights for imbalanced dataset."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    class_weight = np.array(list(class_weights.values()))
    return torch.FloatTensor(class_weight)


def create_random_sample(X, y, sample_size=10000):
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    return X[indices], y[indices]

def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss
    }
    torch.save(checkpoint, path)
    logging.info(f"Model checkpoint saved at {path}")

def load_model_checkpoint(model, optimizer, path , device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    logging.info(f"Model checkpoint loaded from {path} at epoch {epoch}")
    return model, optimizer, epoch, loss_history