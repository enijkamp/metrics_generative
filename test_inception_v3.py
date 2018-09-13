import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from inception_score_v3_torch import inception_score


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


cifar1 = dset.CIFAR10(root='data/', download=True,
                     transform=transforms.Compose([
                         transforms.Scale(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]))

cifar2 = dset.CIFAR10(root='data/', download=True,
                     transform=transforms.Compose([
                         transforms.Scale(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                     ]))

incept1 = inception_score(IgnoreLabelDataset(cifar1), cuda=True, batch_size=1, resize=True, splits=10)
print(incept1)

incept2 = inception_score(IgnoreLabelDataset(cifar1), cuda=True, batch_size=100, resize=True, splits=10)
print(incept2)

incept3 = inception_score(IgnoreLabelDataset(cifar2), cuda=True, batch_size=1, resize=True, splits=10)
print(incept3)

incept4 = inception_score(IgnoreLabelDataset(cifar2), cuda=True, batch_size=100, resize=True, splits=10)
print(incept4)
