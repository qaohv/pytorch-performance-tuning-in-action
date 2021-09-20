import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Normalize

INPUT_SIZE = 256


class PersonSegmentationDataset(Dataset):
    def __init__(self, df_path, root_path, transforms=None, train=True):
        super().__init__()
        self.df = pd.read_csv(df_path)
        self.root_path = root_path
        self.transforms = transforms
        self.train = train

        self.normalizer = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(f"{self.root_path}/{self.df['images'].iloc[idx]}"), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

        if self.train:
            mask = cv2.cvtColor(cv2.imread(f"{self.root_path}/{self.df['masks'].iloc[idx]}"), cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]

            image = self.normalizer(torch.from_numpy(image / 255).float().permute((2, 0, 1)))
            mask = torch.from_numpy(mask[:, :, 0:1] // 255).float().permute((2, 0, 1))

            return image, mask

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        image = self.normalizer(torch.from_numpy(image / 255).float().permute((2, 0, 1)))

        return image
