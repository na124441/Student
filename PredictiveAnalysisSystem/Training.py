
import torch
import pandas as pd
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_PATH = "dataset/train.csv"
VAL_PATH = "dataset/val.csv"

TARGET = "final_exam_score"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PASSING_MARKS = 40

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

print("Training on:", DEVICE)


# ==========================================
# 2. LOAD DATA
# ==========================================

train_data = pd.read_csv(TRAIN_PATH)
val_data = pd.read_csv(VAL_PATH)

# Remove identifiers if present
train_data = train_data.drop(
    columns=["student_id"],
    errors="ignore"
)

val_data = val_data.drop(
    columns=["student_id"],
    errors="ignore"
)

# Separate features and targets
X_train = train_data.drop(columns=[TARGET])
y_train = train_data[TARGET]

X_val = val_data.drop(columns=[TARGET])
y_val = val_data[TARGET]

# Ensure identical feature order
X_val = X_val.reindex(columns=X_train.columns)

# Validate numerical data
if not all(pd.api.types.is_numeric_dtype(dtype)
           for dtype in X_train.dtypes):
    raise ValueError("Non-numerical features detected!")

if X_train.isnull().any().any() or X_val.isnull().any().any():
    raise ValueError("Missing values detected in features!")

if y_train.isnull().any() or y_val.isnull().any():
    raise ValueError("Missing target values detected!")

# Convert to tensors
X_train = torch.tensor(
    X_train.to_numpy(dtype="float32"),
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.to_numpy(dtype="float32"),
    dtype=torch.float32
).unsqueeze(1)

X_val = torch.tensor(
    X_val.to_numpy(dtype="float32"),
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val.to_numpy(dtype="float32"),
    dtype=torch.float32
).unsqueeze(1)


# ==========================================
# 3. CREATE DATASETS
# ==========================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================

class Predictor(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)


input_features = X_train.shape[1]

model = Predictor(input_features).to(DEVICE)

print("Input features:", input_features)
print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))


# ==========================================
# 5. LOSS AND OPTIMIZER
# ==========================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)

            loss = loss_fn(pred, y)

            batch_size = X.size(0)

            total_loss += loss.item() * batch_size

            total_mae += torch.abs(
                pred - y
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mae


# ==========================================
# 7. TRAINING LOOP
# ==========================================

best_val_loss = float("inf")
best_model_state = None

history = {
    "train_loss": [],
    "val_loss": [],
    "val_mae": []
}

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_size = X.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss = total_loss / total_samples

    # Validation
    val_loss, val_mae = evaluate(
        model,
        val_loader
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mae"].append(val_mae)

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_state = deepcopy(
            model.state_dict()
        )

    # Logging
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )


# ==========================================
# 8. RESTORE BEST MODEL
# ==========================================

model.load_state_dict(best_model_state)

print("\nTraining completed!")
print(f"Best Validation MSE: {best_val_loss:.4f}")


# ==========================================
# 9. SAVE MODEL CHECKPOINT
# ==========================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_features": input_features,
    "feature_columns": X_val.shape[1],
    "feature_names": train_data.drop(
        columns=[TARGET]
    ).columns.tolist(),
    "passing_marks": PASSING_MARKS
}

Path("models").mkdir(exist_ok=True)

torch.save(
    checkpoint,
    "models/StudentMarkPredictor.pth"
)

print("Best model saved successfully!")
