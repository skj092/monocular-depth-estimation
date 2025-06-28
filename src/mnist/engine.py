import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, valid_loader=None, epochs=10, learning_rate=0.001, device="cpu", mlflow_log_prefix=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        if mlflow_log_prefix:
            mlflow.log_metric(f"{mlflow_log_prefix}_train_loss", avg_train_loss, step=epoch+1)

        # Validation evaluation
        if valid_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_inputs, val_labels in valid_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    outputs = model(val_inputs)
                    loss = criterion(outputs, val_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == val_labels).sum().item()
                    total += val_labels.size(0)

            avg_val_loss = val_loss / len(valid_loader)
            val_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if mlflow_log_prefix:
                mlflow.log_metric(f"{mlflow_log_prefix}_val_loss", avg_val_loss, step=epoch+1)
                mlflow.log_metric(f"{mlflow_log_prefix}_val_acc", val_acc, step=epoch+1)

def test(model, test_loader, device, mlflow_log_prefix=None, epoch=None):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    if mlflow_log_prefix and epoch is not None:
        mlflow.log_metric(f"{mlflow_log_prefix}_test_accuracy", accuracy, step=epoch+1)

    return accuracy

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
    mlflow_log_prefix=None,
    test_loader=None,
    valid_loader=None  # ← added
):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size(0) * (T**2)
            label_loss = ce_loss(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        if mlflow_log_prefix:
            mlflow.log_metric(f"{mlflow_log_prefix}_train_loss", avg_loss, step=epoch+1)

        # Validation evaluation
        if valid_loader is not None:
            student.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_inputs, val_labels in valid_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    outputs = student(val_inputs)
                    loss = ce_loss(outputs, val_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == val_labels).sum().item()
                    total += val_labels.size(0)

            avg_val_loss = val_loss / len(valid_loader)
            val_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if mlflow_log_prefix:
                mlflow.log_metric(f"{mlflow_log_prefix}_val_loss", avg_val_loss, step=epoch+1)
                mlflow.log_metric(f"{mlflow_log_prefix}_val_acc", val_acc, step=epoch+1)

        # Optional test set evaluation each epoch
        if test_loader is not None:
            acc = test(student, test_loader, device, mlflow_log_prefix=mlflow_log_prefix, epoch=epoch)

