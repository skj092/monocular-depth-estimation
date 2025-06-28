import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
import random
import numpy as np

from engine import train, test, train_knowledge_distillation
from model import DeepNN, LightNN

load_dotenv()

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_PATH"))
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

params = {
    "dataset": "CIFAR10",
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_images_to_keep": 2000,
    "temperature": 2,
    "soft_target_loss_weight": 0.25,
    "ce_loss_weight": 0.75,
    "seed": 42,
}
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

num_images_to_keep = params["num_images_to_keep"]
train_dataset = Subset(train_dataset, range(min(num_images_to_keep, 50_000)))
test_dataset = Subset(test_dataset, range(min(num_images_to_keep, 10_000)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

with mlflow.start_run() as run:
    mlflow.log_params(params)

    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=params["epochs"], learning_rate=params["learning_rate"], device=device, mlflow_log_prefix="teacher")
    acc_deep = test(nn_deep, test_loader, device, mlflow_log_prefix="teacher", epoch=params["epochs"])
    mlflow.pytorch.log_model(nn_deep, "teacher_model")

    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)
    train(nn_light, train_loader, epochs=params["epochs"], learning_rate=params["learning_rate"], device=device, mlflow_log_prefix="student_ce")
    acc_light_ce = test(nn_light, test_loader, device, mlflow_log_prefix="student_ce", epoch=params["epochs"])
    mlflow.pytorch.log_model(nn_light, "student_model_ce")

    torch.manual_seed(42)
    new_nn_light = LightNN(num_classes=10).to(device)
    train_knowledge_distillation(
        teacher=nn_deep,
        student=new_nn_light,
        train_loader=train_loader,
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        T=params["temperature"],
        soft_target_loss_weight=params["soft_target_loss_weight"],
        ce_loss_weight=params["ce_loss_weight"],
        device=device,
        mlflow_log_prefix="student_kd",
        test_loader=test_loader
    )
    acc_light_kd = test(new_nn_light, test_loader, device, mlflow_log_prefix="student_kd", epoch=params["epochs"])
    mlflow.pytorch.log_model(new_nn_light, "student_model_kd")

    mlflow.log_param("teacher_params", sum(p.numel() for p in nn_deep.parameters()))
    mlflow.log_param("student_params", sum(p.numel() for p in nn_light.parameters()))

