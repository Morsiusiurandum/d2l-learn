import torchvision
from torch.utils import data
from torchvision import transforms

from helper import utility


def load_data(batch_size_input: int, workers: int = 4, resize=None):
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    path = utility.get_root_path()+"data"
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size_input, shuffle=True, num_workers=workers),
            data.DataLoader(mnist_test, batch_size_input, shuffle=False, num_workers=workers))
