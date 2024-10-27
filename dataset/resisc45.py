import os
import torch
import abc
import os
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader


ROOT = '' # Set the root directory to the dataset here

# modified from: https://github.com/microsoft/torchgeo
class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and labels at that index
        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    """Abstract base class for classification datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialize a new VisionClassificationDataset instance.
        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        if self.transforms is not None:
            return self.transforms(image), label

        return image, label

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        # label = torch.tensor(label)
        return img, label


class RESISC45(VisionClassificationDataset):
    """RESISC45 dataset.
    The `RESISC45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`_
    dataset is a dataset for remote sensing image scene classification.
    Dataset features:
    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class
    * images extracted from Google Earth from over 100 countries
    * images conditions with high variability (resolution, weather, illumination)
    Dataset format:
    * images are three-channel jpgs
    Dataset classes:
    0. airplane
    1. airport
    2. baseball_diamond
    3. basketball_court
    4. beach
    5. bridge
    6. chaparral
    7. church
    8. circular_farmland
    9. cloud
    10. commercial_area
    11. dense_residential
    12. desert
    13. forest
    14. freeway
    15. golf_course
    16. ground_track_field
    17. harbor
    18. industrial_area
    19. intersection
    20. island
    21. lake
    22. meadow
    23. medium_residential
    24. mobile_home_park
    25. mountain
    26. overpass
    27. palace
    28. parking_lot
    29. railway
    30. railway_station
    31. rectangular_farmland
    32. river
    33. roundabout
    34. runway
    35. sea_ice
    36. ship
    37. snowberg
    38. sparse_residential
    39. stadium
    40. storage_tank
    41. tennis_court
    42. terrace
    43. thermal_power_station
    44. wetland
    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:
    * https://arxiv.org/abs/1911.06721
    If you use this dataset in your research, please cite the following paper:
    * https://doi.org/10.1109/jproc.2017.2675998
    """

    # url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
    # md5 = "d824acb73957502b00efd559fc6cfbbb"
    # filename = "NWPU-RESISC45.rar"
    directory = "resisc45/NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new RESISC45 dataset instance.
        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        assert split in self.splits
        self.root = root

        valid_fns = set()
        with open(os.path.join(self.root, "resisc45", f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(
            x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )


def prepare_train_loaders(config):
    train_dataset = RESISC45(
        split='train',
        transforms=config['train_preprocess'],
        root=ROOT,
    )
    loaders = {
        'full': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
    }
    return loaders

def prepare_test_loaders(config):
    test_dataset = RESISC45(
        split='test',
        transforms=config['eval_preprocess'],
        root=ROOT,
    )
    
    val_dataset = RESISC45(
        split='val',
        transforms=config['eval_preprocess'],
        root=ROOT,
    )
    
    loaders = {
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        ),
    }
    loaders['class_names'] = [' '.join(c.split('_')) for c in RESISC45.classes]
    
    return loaders