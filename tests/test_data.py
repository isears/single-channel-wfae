from scwfae.data.ptbxlDS import PtbxlAllChanDS, PtbxlDS
from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
import pytest


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module", autouse=True)
def setup_module_level():
    print("Setting up data tests...")

    torch.manual_seed(42)
    ds = PtbxlDS(lowres=True)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds))

    if not Path.exists(DATA_DIR / "validation_indices"):
        with open(DATA_DIR / "validation_indices", "w") as f:
            f.writelines([str(i) + "\n" for i in val_dl.dataset.indices])  # type: ignore


def test_consistent_validation():
    torch.manual_seed(42)
    ds = PtbxlAllChanDS(lowres=True)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds))

    with open(DATA_DIR / "validation_indices", "r") as f:
        validation_indices = [int(i) for i in f.readlines()]

    for i in val_dl.dataset.indices:  # type: ignore
        assert i in validation_indices
