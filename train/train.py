import warnings
import logging
import yaml

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from src.utils import train, run_inference
from src.dataset import ImageRegressionDataset
from src.models import build_custom_resnet18, EfficientNetRegression


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


drivnet_csv_path = "./DrivAerNetPlusPlus_Drag_8k.csv"
drivenet_img = "images"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset_drive = ImageRegressionDataset(drivnet_csv_path, drivenet_img, transform=image_transforms)
train_size = int(0.8 * len(dataset_drive))
test_size = len(dataset_drive) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_drive, [train_size, test_size])

model_1 = build_custom_resnet18(in_channels=9)
train(model_1, train_dataset, test_dataset, config["resnet18"])

model_2 = EfficientNetRegression()
train(model_2, train_dataset, test_dataset, config["efficientnet"])

dataloader = DataLoader(dataset_drive, batch_size=16, shuffle=False)

predictions_1 = run_inference(model_1.to("cuda"), dataloader, device="cuda")
predictions_2 = run_inference(model_2.to("cuda"), dataloader, device="cuda")

targets = [label for _, label in dataset_drive.pairs]

df = pd.DataFrame({
    "resnet18_Prediction": predictions_1,
    "effnetb1full_Prediction": predictions_2,
    'Target': targets
})
df.to_csv("model_predictions_norm.csv", index=False)


# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 300  # 5 minutes
TARGET_NAME = "Target"

# Load dataset
data = df

# Split dataset into train and test sets
train_data, test_data = train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

logging.info(f"Data split completed. Train data size: {train_data.shape}, Test data size: {test_data.shape}")

# Define roles
roles = {
    "target": TARGET_NAME  # Specify the target column
}

# Define the task
task = Task("reg")

# Initialize the AutoML pipeline
automl = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE}
)

# Train the AutoML model
logging.info("Starting AutoML training...")
out_of_fold_predictions = automl.fit_predict(train_data, roles=roles, verbose=1)

# Make predictions on the test set
logging.info("Predicting on test data...")
test_predictions = automl.predict(test_data)

# Extract predictions
predicted_values = test_predictions.data[:, 0]

# Evaluate the model
mse = mean_squared_error(test_data[TARGET_NAME], predicted_values)
mae = mean_absolute_error(test_data[TARGET_NAME], predicted_values)
r2 = r2_score(test_data[TARGET_NAME], predicted_values)

logging.info(f"Mean Squared Error (MSE): {mse}")
logging.info(f"Mean Absolute Error (MAE): {mae}")
logging.info(f"R^2 Score: {r2}")
