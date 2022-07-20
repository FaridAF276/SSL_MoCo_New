from gauss_blur import GaussianBlur
from generate_crops import TwoCropsTransform
import torchvision
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
#from exceptions.exceptions import InvalidDatasetSelection
class FolderPair(datasets.ImageFolder):
    def __getitem__(self, index):
        # img = self.data[index]
        path = self.imgs[index][0]
        img = Image.open(path)
        #Since some images are grayscale and others and RGB we convert grayscale images to RGB
        if img.mode is not "RGB":
          img = img.convert('RGB')
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2
class MocoDatasetGenerator:
    def __init__(self, root_folder='./data'):
        self.root_folder = root_folder
        self.test_transform = transforms.Compose([\
                              transforms.ToTensor(),\
                              transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    def get_moco_transformation_pipeline(self, size, aug_plus=False, chest=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if chest:
            augmentation = [
                transforms.RandomRotation(degrees=(0,180), expand=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.Resize(size=(224,224)),
                transforms.ToTensor(),
                normalize
            ]
            return transforms.Compose(augmentation)
        if aug_plus:
            augmentation = [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)), #Pas pertinent pour ChestXray
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), #Pas partinent pour Chestxray
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            augmentation = [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        data_transform = transforms.Compose(augmentation)
        return data_transform

    def get_moco_dataset(self, dataset_name, train_root=''):
        print(dataset_name)
        dataset_dictionary = {
            'cifar10': 'datasets.CIFAR10(root = self.root_folder, train=True,transform=TwoCropsTransform(self.get_moco_transformation_pipeline(size=32, aug_plus = False)),download=True)',
            'stl10': 'datasets.STL10(root = self.root_folder, split="unlabeled", transform=TwoCropsTransform(self.get_moco_transformation_pipeline(size=96, aug_plus=False)), download=True)',
            'mnist': 'datasets.MNIST(root = self.root_folder, train=True, transform=TwoCropsTransform(self.get_moco_transformation_pipeline(size=28, aug_plus=False)), download=True)',
            'folder': 'FolderPair(root=train_root, transform=TwoCropsTransform(self.get_moco_transformation_pipeline(size=512, aug_plus=True, chest=False)))'
            }
        try:
            dataset_fn = dataset_dictionary[dataset_name]  # lambda fn
        except KeyError:
            print("temp exception")#raise InvalidDatasetSelection()  # return lambda_fn returns the object`
        else:
            return eval(dataset_fn)

    def get_moco_data_loader(self, dataset_name, batch_size, test_root=''):
        if(dataset_name == "cifar10"):
            memory_data = datasets.CIFAR10(root=self.root_folder, train=True, transform=self.test_transform, download=True)
            memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_data = datasets.CIFAR10(root='data', train=False, transform=self.test_transform, download=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        elif(dataset_name == "stl10"):
            memory_data = datasets.STL10(root=self.root_folder, split='train', transform=self.test_transform, download=True)
            memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_dataset = datasets.STL10(root=self.root_folder, split='test', download=True, transform=self.test_transform)                         
            test_loader = DataLoader(test_dataset, batch_size=2*batch_size,num_workers=10, drop_last=False, shuffle=False)
        elif(dataset_name =='folder'):
            memory_data = datasets.ImageFolder(root=test_root, transform=self.test_transform)
            memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
            test_data = datasets.ImageFolder(root=test_root, transform=self.test_transform)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
        return memory_loader, test_loader
                                    
        

   
        