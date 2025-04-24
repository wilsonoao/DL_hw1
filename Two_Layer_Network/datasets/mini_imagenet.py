from torch.utils.data import Dataset
from PIL import Image
import os

class MiniImageNet(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
