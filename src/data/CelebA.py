import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataset(dsize=(64, 64), root="/mnt/mfs/yiling/new_EL_surface/train"):
    if isinstance(dsize, int):
        dsize=(dsize, dsize)
    transform = T.Compose([
        T.Resize(dsize),
        T.ToTensor(),
        # T.Lambda(lambda x:x*2-1)
    ])
    dataset = torchvision.datasets.ImageFolder(root, transform)
    return dataset