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
# (9.672783335442967, 0.14991609945479523)
assert incept1[0] > 9.5

incept2 = inception_score(IgnoreLabelDataset(cifar1), cuda=True, batch_size=10, resize=True, splits=10)
print(incept2)
# (9.67278277984201, 0.14991598186787766)
assert incept2[0] > 9.5

incept3 = inception_score(IgnoreLabelDataset(cifar2), cuda=True, batch_size=1, resize=True, splits=10)
print(incept3)
# (10.45415392273974, 0.1428910206001064)
assert incept3[0] > 10.2

incept4 = inception_score(IgnoreLabelDataset(cifar2), cuda=True, batch_size=10, resize=True, splits=10)
print(incept4)
# (10.4541538553588, 0.14289089976856756)
assert incept4[0] > 10.2
