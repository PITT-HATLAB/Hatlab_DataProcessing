from typing import Any, Union, Optional, Dict, Type, Collection
from pathlib import Path
import time
from contextlib import nullcontext

import yaml
import numpy as np

from plottr.data import DataDict, MeshgridDataDict
from plottr.data.datadict_storage import datadict_to_hdf5, DDH5Writer, DATAFILEXT


class HatDDH5Writer(DDH5Writer):
    """Context manager for writing data to DDH5.
    Based on the DDH5Writer from plottr, with re-implemented data_folder method, which allows user to specify the name
    of the folder, instead of using a generated ID.

    :param datadict: Initial data object. Must contain at least the structure of the
        data to be able to use :meth:`add_data` to add data.
    :param basedir: The root directory in which data is stored.
    :param foldername: name of the folder where the data will be stored. If not provided, the current date will be used
        as the foldername.
    :param filename: Filename to use. Defaults to 'data.ddh5'.
    :param groupname: Name of the top-level group in the file container. An existing
        group of that name will be deleted.
    :param name: Name of this dataset. Used in path/file creation and added as meta data.

    """

    def __init__(self, datadict: DataDict, basedir: str = '.', foldername: Optional[str] = None, filename: str = 'data',
                 groupname: str = 'data', name: Optional[str] = None):
        super().__init__(datadict, basedir, groupname, name, filename)
        self.foldername = foldername

    def data_folder(self) -> Path:
        """Return the folder
        """
        if self.foldername is None:
            path = Path(time.strftime("%Y-%m-%d"), )

        else:
            path = Path(self.foldername)
        return path

    def save_config(self, cfg:Dict):
        datafolder = str(self.filepath.parent)
        with open(str(datafolder) + f"\\\\{str(self.filename)}_cfg.yaml", 'w') as file:
            yaml.dump(cfg, file)

    def data_file_path(self) -> Path:
        """Instead of checking for duplicate folders, here we check for duplicate filenames, so that we can have
        so that we can have different files in the same date folder.

        :returns: The filepath of the data file.
        """
        data_file_path = Path(self.basedir, self.data_folder(), str(self.filename)+f".{DATAFILEXT}")
        appendix = ''
        idx = 2
        while data_file_path.exists():
            appendix = f'-{idx}'
            data_file_path = Path(self.basedir,
                                    self.data_folder(),  str(self.filename)+appendix+f".{DATAFILEXT}")
            idx += 1
        self.filename = Path(str(self.filename)+appendix)
        return data_file_path

class DummyWriter(nullcontext):
    def __init__(self):
        super().__init__()
    def save_config(self, *args, **kwargs):
        pass
    def add_data(self, *args, **kwargs):
        pass



if __name__=="__main__":

    a = np.zeros((10, 50))
    for i in range(len(a)):
        a[i] = np.linspace(i, i + 10, 50)
    xlist = np.arange(10)
    ylist = np.arange(50) * 2

    data = {
        "a": {
            "axes": ["x", "y"],
            "unit": "a.u."
        },
        "x": {
            'axes': []
        },
        "y": {
            'axes': []
        }
    }
    dd = DataDict(**data)
    ddw = HatDDH5Writer(dd, r"L:\Data\SNAIL_Pump_Limitation\test\\", foldername=None, filename="data11")
    # ddw = DDH5Writer(dd, r"L:\Data\SNAIL_Pump_Limitation\test\\", filename="data11", name="testnam")
    with ddw as d:
        for i in range(5):
            for j in range(5):
                d.add_data(
                    a=a[i, j],
                    x=xlist[i],
                    y=ylist[j]
                )
        d.save_config({"a":2})


    # inner_sweeps = DataDict(amp={"unit": "DAC", "values": [1,2,3]})
    # outer_sweeps = DataDict(freq={"unit": "MHz", "values": [4,5,6]})
    # qdd = QickDataDict([0,1], inner_sweeps, outer_sweeps=outer_sweeps)
    # qddw = hatDDH5Writer(qdd, r"L:\Data\SNAIL_Pump_Limitation\sweepSNAILPumpAmpFreq\sunHarmonic\\",
    #                     foldername="08162022", filename=f"Q3_test1")
    # with qddw as d:
    #     for i in range(10):
    #         for j in range(5):
    #             d.add_data(inner_sweeps=0, avg_i=0, avg_q=0, freq=1)

