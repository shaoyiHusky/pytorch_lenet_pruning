import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.generateMask import lenet_prune
from pruning.utils import train, test, prune_rate
from models import LeNet


# Hyper Parameters
param = {
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}

train_dataset = datasets.MNIST(root='../data/', train=True, download=True, transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
test_dataset = datasets.MNIST(root='../data/', train=False, download=True, transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=param['test_batch_size'], shuffle=True)

model = LeNet()
model.load_state_dict(torch.load('models/lenet_pretrained.pkl', map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("--- Accuracy of Pretrained Model ---")
test(model, loader_test)

# pruning
masks = lenet_prune()
model.set_masks(masks)
print("--- Accuracy After Pruning ---")
test(model, loader_test)

# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
train(model, criterion, optimizer, param, loader_train)

print("--- Accuracy After Retraining ---")
test(model, loader_test)
prune_rate(model)

# Save and load the entire model
torch.save(model.state_dict(), 'models/lenet_pruned.pkl')
