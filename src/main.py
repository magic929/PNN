from torch.utils.data import DataLoader
import sys
sys.path.append("./src")

from utils.data_preprocess import Data
from models import PNN1


def train():
    dataset = Data("./input/3358/train.txt", "./input/3358/featindex.txt")
# loader = DataLoader(dataset, 128, True, num_workers=4)

    pnn1 = PNN1(dataset.field_sizes, dataset.feature_sizes)
    pnn1.fit(dataset, save_path="./output/model.pk")

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    if sys.argv[2] == "test":
        