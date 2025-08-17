import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from model import FishClassifier
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from collections import deque

def main():
    # Veri hazırlama ve artırma
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root="NA_Fish_Dataset", transform=transform)

    # Eğitim ve test kümelerine bölme
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modeli oluşturma ve eğitme
    model = FishClassifier(num_classes=len(dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5  # Epoch sayısı 5 olarak ayarlandı

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Model doğrulama
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")

    # Eğitilen modelin kaydedilmesi
    torch.save(model.state_dict(), "custom_fish_classifier.pth")

    # OpenCV ile gerçek zamanlı görüntü işleme ve sınıflandırma
    cap = cv2.VideoCapture(0)  # Kamera başlat
    class_names = dataset.classes  # Sınıf isimlerini kontrol edin

    # Tahminleri saklamak için bir tampon oluşturun
    prediction_buffer = deque(maxlen=30)  # Son 30 tahmini sakla
    prediction_threshold = 70.0  # Eşik değeri, yüzde

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntü işleme
        resized_frame = cv2.resize(frame, (128, 128))
        img_array = np.array(resized_frame, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)

        # Tahmin yap
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence_percentage = confidence.item() * 100
            if predicted.item() < len(class_names):  # IndexError kontrolü
                predicted_class = class_names[predicted.item()]
            else:
                predicted_class = "Unknown"
            
            # Eğer güven oranı eşik değerinin üzerindeyse tahminleri tampona ekle
            if confidence_percentage > prediction_threshold:
                prediction_buffer.append((predicted_class, confidence_percentage))

        # En yüksek doğruluk oranına sahip tahmini seç
        best_prediction = max(prediction_buffer, key=lambda x: x[1], default=("None", 0))

        # Sonuçları görüntüde göster
        if best_prediction[1] >= prediction_threshold:
            cv2.putText(frame, f"{best_prediction[0]} ({best_prediction[1]:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Fish Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()