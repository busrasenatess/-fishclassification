import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import FishClassifier

def load_model(model_path, num_classes):
    model = FishClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy:.2f}")

    # Kesinlik grafiği
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, precision, color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Classes')
    plt.ylabel('Precision')
    plt.title('Precision per Class')
    plt.show()

    # Duyarlılık grafiği
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, recall, color='green')
    plt.ylim(0, 1)
    plt.xlabel('Classes')
    plt.ylabel('Recall')
    plt.title('Recall per Class')
    plt.show()

    # F1 skoru grafiği
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, f1, color='red')
    plt.ylim(0, 1)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.show()

    # Doğruluk grafiği
    plt.figure(figsize=(6, 6))
    plt.bar(['Accuracy'], [accuracy], color='purple')
    plt.ylim(0, 1)
    plt.title('Overall Accuracy')
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root="NA_Fish_Dataset", transform=transform)
    test_size = int(0.2 * len(dataset))
    test_dataset, _ = random_split(dataset, [test_size, len(dataset) - test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_path = "custom_fish_classifier.pth"
    model = load_model(model_path, num_classes=len(dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, dataset.classes)
    plot_metrics(y_true, y_pred, dataset.classes)
