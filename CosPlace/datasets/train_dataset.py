
import os
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_image(path):
    return Image.open(path).convert("RGB")


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder, M=10, alpha=30, N=5, L=2,
                 current_group=0, min_images_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_images_per_class : int, minimum number of image in a class.
        """
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.N = N
        self.L = L
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.augmentation_device = args.augmentation_device
        
        # dataset_name should be either "processed", "small" or "raw", if you're using SF-XL
        dataset_name = os.path.basename(args.dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, alpha, L, min_images_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        classes_per_group, self.images_per_class = torch.load(filename)
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups by setting for example " +
                             f"'--groups_num {current_group}'")
        self.classes_ids = classes_per_group[current_group]
        
        if self.augmentation_device == "cpu":
            self.transform = T.Compose([
                    T.ColorJitter(brightness=args.brightness,
                                  contrast=args.contrast,
                                  saturation=args.saturation,
                                  hue=args.hue),
                    T.RandomResizedCrop([512, 512], scale=[1-args.random_resized_crop, 1]),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    
    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        
        class_id = self.classes_ids[class_num]
        # Pick a random image among those in this class.
        image_path = random.choice(self.images_per_class[class_id])
        
        try:
            pil_image = open_image(image_path)
        except Exception as e:
            logging.info(f"ERROR image {image_path} couldn't be opened, it might be corrupted.")
            raise e
        
        tensor_image = T.functional.to_tensor(pil_image)
        assert tensor_image.shape == torch.Size([3, 512, 512]), \
            f"Image {image_path} should have shape [3, 512, 512] but has {tensor_image.shape}."
        
        if self.augmentation_device == "cpu":
            tensor_image = self.transform(tensor_image)
        
        return tensor_image, class_num, image_path
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])
    
    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    @staticmethod
    def initialize(dataset_folder, M, N, alpha, L, min_images_per_class, filename):
        logging.debug(f"Searching training images in {dataset_folder}")
        
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        logging.debug(f"Found {len(images_paths)} images")
        
        logging.debug("For each image, get its UTM east, UTM north and heading from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float)
        
        logging.debug("For each image, get class and group to which it belongs")
        class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, alpha, N, L)
                              for m in utmeast_utmnorth_heading]
        
        logging.debug("Group together images belonging to the same class")
        images_per_class = defaultdict(list)
        for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
            images_per_class[class_id].append(image_path)
        
        # Images_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}
        
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, images_per_class), filename)
    
    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, heading, M, alpha, N, L):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north and
            heading (e.g. (396520, 4983800,120)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        rounded_heading = int(heading // alpha * alpha)
        
        class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)
        # group_id goes from (0, 0, 0) to (N, N, L)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M,
                    rounded_heading % (alpha * L) // alpha)
        return class_id, group_id
