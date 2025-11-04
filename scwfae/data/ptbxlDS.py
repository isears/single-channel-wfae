import torch
import pandas as pd
from scwfae.data import load_single_ptbxl_record
import ast
import numpy as np
import random
import neurokit2 as nk


class PtbxlDS(torch.utils.data.Dataset):
    channel_names = [
        "I",
        "II",
        "III",
        "AVF",
        "AVL",
        "AVR",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    def __init__(
        self,
        root_folder: str = "./data/ptbxl",
        lowres: bool = False,
    ):
        """Base PTBXL dataset initialization

        Args:
            root_folder (str, optional): Path to PTBXL data. Defaults to "./data/ptbxl".
            lowres (bool, optional): Whether to use the 100Hz (True) or 500Hz (False) data. Defaults to False.
            return_labels (bool, optional): Whether to return diagnostic labels for each EKG. Label returning was made optional because it is not necessary for the autoencoder training loop and will probably slow down the dataloaders significantly. Defaults to False.

        Returns:
            None: None
        """
        super(PtbxlDS, self).__init__()

        random.seed(42)

        metadata = pd.read_csv(f"{root_folder}/ptbxl_database.csv")
        metadata.ecg_id = metadata.ecg_id.astype(int)
        metadata.patient_id = metadata.patient_id.astype(int)
        self.metadata = metadata
        self.patient_ids = self.metadata["patient_id"].unique()
        self.pid_groups = self.metadata.groupby("patient_id")

        self.lowres = lowres
        self.root_folder = root_folder

        # Get PTBXL labels
        self.metadata.scp_codes = self.metadata.scp_codes.apply(ast.literal_eval)

        # Modified from physionet example.py
        scp_codes = pd.read_csv(f"{root_folder}/scp_statements.csv", index_col=0)
        scp_codes = scp_codes[scp_codes.diagnostic == 1]

        self.ordered_labels = list()

        for diagnostic_code, description in zip(scp_codes.index, scp_codes.description):
            self.ordered_labels.append(description)
            self.metadata[description] = self.metadata.scp_codes.apply(
                lambda x: diagnostic_code in x.keys()
            ).astype(float)

    def __len__(self):
        return len(self.pid_groups)

    def __getitem__(self, index: int):
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        patient_id = self.patient_ids[index]
        available_exams = self.pid_groups.get_group(patient_id)
        selected_exam = available_exams.sample(1, random_state=42).iloc[0]
        selected_channel = random.choice(self.channel_names)

        ecg_id = selected_exam["ecg_id"]
        sig, sigmeta = load_single_ptbxl_record(
            ecg_id, selected_channel, lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = nk.ecg_clean(sig.squeeze(), sampling_rate=sigmeta["fs"])  # type: ignore
        # Create a contiguous copy to avoid negative stride issues
        sig_clean = np.ascontiguousarray(sig_clean)

        labels = {c: bool(selected_exam[c].item()) for c in self.ordered_labels}
        labels["channel"] = int(self.channel_names.index(selected_channel))  # type: ignore

        return torch.Tensor(sig_clean).float(), labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = PtbxlDS(lowres=False)
    dl = DataLoader(dataset=ds, batch_size=32)

    for batch in dl:
        print(batch)
        break
