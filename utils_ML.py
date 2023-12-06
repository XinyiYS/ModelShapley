import torch
import torch.nn as nn

def train(model, device, train_loader, optimizer, E, scheduler=None, verbose=False, criterion=nn.CrossEntropyLoss()):

    model.train()
    model = model.to(device)
    for e in range(E):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if verbose and (e+1) % 10 == 0:
            print('Current epoch is : {} with training loss: {}'.format(str(e+1), str(loss.item()) )  )

        if scheduler:
            scheduler.step()

    return model


def test(model, device, test_loader, verbose=False, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    model = model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)