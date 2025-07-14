import sys, os

import torch
import os
from torch import nn
from dam4sam_tracker import DAM4SAMTracker
from typing import Tuple


def export_predict(tracker_name="sam21pp-L") -> Tuple[nn.Module, DAM4SAMTracker]:
    tracker = DAM4SAMTracker(tracker_name)
    module: nn.Module = tracker.predictor
    return module, tracker


if __name__ == "__main__":
    predict,_ = export_predict("sam21pp-L")
    print(isinstance(predict, nn.Module))
