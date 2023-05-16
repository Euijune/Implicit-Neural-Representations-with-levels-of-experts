import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
    
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import skimage

class ImageFitting(Dataset):
    def __init__(self, img_path, sidelength=256, mgrid_min=0, mgrid_max=1, normalize=False):
        super().__init__()
        self.sidelen = sidelength
        self.mgrid_min = mgrid_min
        self.mgrid_max = mgrid_max
        self.normalize = normalize

        self.img = self.get_image_tensor(img_path)
        self.pixels = self.img.permute(1, 2, 0).view(-1, 1)
        self.coords = self.get_mgrid(dim=2)
    
    def get_mgrid(self, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of mgrid_min to mgrid_max.
        sidelen: int
        dim: int'''
        tensors = tuple(dim * [torch.linspace(self.mgrid_min, self.mgrid_max, steps=self.sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid
    
    def get_image_tensor(self, img_path):
        if img_path == 'cameraman':
            img = Image.fromarray(skimage.data.camera())
            self.normalize = True
        else:
            img = Image.open(img_path).convert("L")        
        
        transform_list = [
            Resize(self.sidelen),
            ToTensor()
        ]
        if self.normalize:
            transform_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))
        
        transform = Compose(transform_list)
        img = transform(img)

        return img

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels