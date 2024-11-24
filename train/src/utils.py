
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from accelerate import Accelerator


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def train(model, train_dataset, test_dataset, config):
    accelerator = Accelerator()
    device = accelerator.device

    model = torch.nn.DataParallel(model)#, device_ids = [0,1])
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=150, num_workers=4, shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=150, num_workers=4, shuffle=False, pin_memory=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)  

    model, optimizer, data, test_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, scheduler)

    name = config["name"]
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            train_progress_bar.set_postfix(loss=loss.item())
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        if epoch % 3 == 0:
            model.eval()
            predictions, true_labels = [], []

            # Add tqdm progress bar for the validation loop
            validation_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
            with torch.no_grad():
                for images, labels in validation_progress_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).squeeze()
                    predictions.extend(outputs.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            mse = mean_squared_error(true_labels, predictions)
            r2 = r2_score(true_labels, predictions)
            correlation = np.corrcoef(true_labels, predictions)[0, 1]
            logging.info(f"Validation MSE: {mse:.6f}, R2: {r2:.6f}, Correlation: {correlation:.6f}")
            
            model_name = f"model{name}{r2:.6f}_{correlation:.6f}.pth"
            # torch.save(model.state_dict(), model_name)
            logging.info(f"Model saved successfully as {model_name}")


def run_inference(model, dataloader, device="cuda"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
    return predictions
