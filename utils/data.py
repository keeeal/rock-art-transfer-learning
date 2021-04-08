
from random import random, choice, sample

from torch import get_num_threads
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


normalise = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=20, translate=(0.1, 0.1), shear=10,
        # interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.RandomResizedCrop(
        size=(224, 224), scale=(0.8, 1.2), ratio=(0.9, 1.1),
    ),
])

augment_and_normalise = transforms.Compose([augment, normalise])


# wraps an image folder, provides methods to get matching and differing samples
class SiameseDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.imgs = ImageFolder(root, transform)
        self.targets = {k:[] for k in set(self.imgs.targets)}
        for n, k in enumerate(self.imgs.targets): self.targets[k].append(n)

    def __repr__(self):
        return repr(self.imgs)

    def __getitem__(self, index):
        i, j = self._unravel(index)
        x_i, x_j = self.imgs[i], self.imgs[j]
        return x_i[0], x_j[0], float(x_i[1] == x_j[1])

    def __len__(self):
        return len(self.imgs)**2

    def _ravel(self, i, j):
        return i * len(self.imgs) + j

    def _unravel(self, index):
        return index // len(self.imgs), index % len(self.imgs)

    def get_matching_idx(self):
        i, j = (choice(self.targets[k]) for k in 2*[choice(list(self.targets))])
        return self._ravel(i, j)

    def get_differing_idx(self):
        i, j = (choice(self.targets[k]) for k in sample(list(self.targets), 2))
        return self._ravel(i, j)


# chooses matching and differing samples equally often with replacement
class SiameseRandomSampler(Sampler):
    def __init__(self, data, num_samples=None):
        super().__init__(data)
        self.data = data
        self.num_samples = num_samples if num_samples else len(self.data)

    def __iter__(self):
        return iter([self.data.get_matching_idx() if random() < .5
            else self.data.get_differing_idx() for _ in range(self.num_samples)])

    def __len__(self):
        return self.num_samples


# return the data loader of a siamese dataset 
def load_siamese(root, transform, batch_size=1, batches=0):
    data = SiameseDataset(root, transform)
    num_samples = batches * batch_size if 0 < batches else len(data.imgs)
    sampler = SiameseRandomSampler(data, num_samples=num_samples)
    loader = DataLoader(data, batch_size, sampler=sampler, num_workers=get_num_threads())
    return loader


# return the data loader of an image folder
def load(root, transform, batch_size=1, batches=0):
    data = ImageFolder(root, transform)
    num_samples = batches * batch_size if 0 < batches else len(data.imgs)
    sampler = RandomSampler(data, replacement=True, num_samples=num_samples)
    loader = DataLoader(data, batch_size, sampler=sampler, num_workers=get_num_threads())
    return loader
