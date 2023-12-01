import torch

from utils.utils import create_dataloader, create_dataset, prepare_model


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = prepare_model()
model = model.to(device)

train_dataloader = create_dataloader(
    is_train=True, 
    batch_size=1024,
    shuffle=True,
    )
test_dataloader = create_dataloader(
    is_train=False,
    batch_size=1024,
    shuffle=False,
)
optimizer = torch.optim.Adam(model.parameters(), 0.0001)
criterion = torch.nn.CrossEntropyLoss().to(device)

test_data_num = len(create_dataset(is_train=False))
epochs = 500
max_accuracy = 0.0
for epoch in range(epochs):
    model.train()
    loss_sum = 0
    for images, labels in train_dataloader:
        model.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        loss_sum += loss
        loss.backward()
        optimizer.step()
    model.eval()
    correct_num = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        pred_label = torch.argmax(preds, dim=1)
        correct_num += torch.sum(labels == pred_label)
    loss_ave = loss_sum / len(train_dataloader)
    accuracy = correct_num / test_data_num
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
        }, "./best.pth")
        print(f"Epoch: {epoch}, Train Loss: {round(loss_ave.item(), 3)}, Val Accuracy: {round(accuracy.item(), 3)}, Model updated!")
        continue
    print(f"Epoch: {epoch}, Train Loss: {round(loss_ave.item(), 3)}, Val Accuracy: {round(accuracy.item(), 3)}")
