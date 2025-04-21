import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Noise(nn.Module):
    def __init__(self, *dim):
        super(Noise, self).__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)

    def forward(self):
        return self.noise
    
def data_prepare(classes, train_ds, valid_ds, classes_to_forget, batch_size):
    num_classes = len(classes)

    classwise_train = {}
    for i in range(num_classes):
        classwise_train[i] = []
    
    for img, label in train_ds:
        classwise_train[label].append((img, label))

    classwise_test = {}
    for i in range(num_classes):
        classwise_test[i] = []

    for img, label in valid_ds:
        classwise_test[label].append((img, label))

    num_samples_per_class = 3000

    retain_samples = []
    for i in range(len(classes)):
        if classes[i] not in classes_to_forget:
            retain_samples += classwise_train[i][:num_samples_per_class]

    retain_valid = []
    for cls in range(num_classes):
        if cls not in classes_to_forget:
            for img, label in classwise_test[cls]:
                retain_valid.append((img, label))

    forget_valid = []
    for cls in range(num_classes):
        if cls in classes_to_forget:
            for img, label in classwise_test[cls]:
                forget_valid.append((img, label))

    forget_valid_dl = torch.utils.data.DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)
    retain_valid_dl = torch.utils.data.DataLoader(retain_valid, batch_size * 2, num_workers=3, pin_memory=True)

    return retain_samples, forget_valid_dl, retain_valid_dl

def train_noise(model, retain_samples, classes_to_forget, batch_size):
    noises = {}
    print("-------------TRAIN NOISE-----------------")
    for cls in classes_to_forget:
        print(f"Optiming loss for class{cls}")
        noises[cls] = Noise(batch_size, 3, 32, 32).to(device)
        opt = torch.optim.Adam(noises[cls].parameters(), lr = 0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]()
                labels = torch.zeros(batch_size).to(device) + class_label
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            print(f"Loss: {np.mean(total_loss)}")
    noisy_data = []
    num_batches = 20
    class_num = 0

    for cls in classes_to_forget:
        for i in range(num_batches):
            batch = noises[cls]().cpu().detach()
            for i in range(batch[0].size(0)):
                noisy_data.append((batch[i], torch.tensor(class_num)))

    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))

    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size, shuffle=True)
    return noisy_loader, other_samples

def impair(model, train_ds, noisy_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.02)
    print("---------------------IMPAIR----------------------")
    for epoch in range(1):
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim = 1)
            assert out.shape == labels.shape
            running_acc += (labels == out).sum().item()
        print(f"Epoch = {epoch + 1} | Train Loss = {running_loss / len(train_ds)} | Train Acc = {running_acc * 100 / len(train_ds)}%")


def repair(model, train_ds, other_samples, batch_size):
    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    print("---------------------REPAIR----------------------")
    
    for epoch in range(10):
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim = 1)
            assert out.shape == labels.shape
            running_acc += (labels == out).sum().item()
        print(f"Epoch = {epoch + 1} | Train Loss = {running_loss / len(train_ds)} | Train Acc = {running_acc * 100 / len(train_ds)}%")

def validate(model, valid_dl):
    model.eval()
    output = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = F.cross_entropy(out, labels)
            _, preds = torch.max(out, dim = 1)
            acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
            output.append({"Loss": loss, "Acc": acc})
    batch_losses = [X['Loss'] for X in output]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_acc = [X['Acc'] for X in output]
    epoch_acc = torch.stack(batch_acc).mean()
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}