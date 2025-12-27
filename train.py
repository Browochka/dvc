import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import mlflow
import json
from sklearn.model_selection import train_test_split


with open("/Users/browka/Documents/squirell/params.yaml") as f:
    params = yaml.safe_load(f)

data_dir = params["data"]["data_dir"]
batch_size = params["training"]["batch_size"]
epochs = params["training"]["epochs"]
lr = params["training"]["lr"]
test_size = params["training"]["test_size"]
num_classes = params["model"]["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)


trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
testTransform = transforms.Compose([
    transforms.Resize(int(224*1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


full_dataset = datasets.ImageFolder(data_dir)
all_labels = [label for _, label in full_dataset.samples]

train_indices, test_indices = train_test_split(
    range(len(full_dataset)),
    test_size=test_size,
    random_state=42,
    stratify=all_labels
)

train_dataset = Subset(
    datasets.ImageFolder(data_dir, transform=trainTransform),
    train_indices
)
test_dataset = Subset(
    datasets.ImageFolder(data_dir, transform=testTransform),
    test_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


class SquirrelEfficientNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(SquirrelEfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.features[-5:].parameters():
            param.requires_grad = True
        
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

model = SquirrelEfficientNet(num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)


mlflow.set_experiment("squirrel_classification")
with mlflow.start_run():
    mlflow.log_params({
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "num_classes": num_classes
    })


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch+1)


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")
    mlflow.log_metric("val_accuracy", acc)
    
    metrics = {
    "train_loss": avg_loss,
    "val_accuracy": acc,
    "epochs": epochs
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)



    model_path = "/Users/browka/Documents/squirell/models/SquirrelEff.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
