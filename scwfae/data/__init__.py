import wfdb
from typing import Optional


def load_single_ptbxl_record(
    id: int,
    channel: Optional[str] = None,
    lowres: bool = True,
    root_dir: str = "./data/ptb-xl",
):
    id_str = "{:05d}".format(id)
    prefix_dir = f"{id_str[:2]}000"

    if lowres:
        load_path = f"{root_dir}/records100/{prefix_dir}/{id_str}_lr"
    else:
        load_path = f"{root_dir}/records500/{prefix_dir}/{id_str}_hr"

    if channel:
        return wfdb.rdsamp(load_path, channel_names=[channel])
    else:
        return wfdb.rdsamp(load_path)
