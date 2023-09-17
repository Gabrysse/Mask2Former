#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained swin model:
  gdown 1EQpZch2NCswzGVUvrbwjg_qG4QYeyiIU
  # run the conversion
  ./convert-pretrained-stdc-model-to-d2.py STDCNet813M_73.91.tar STDCNet813M_73.91.pkl
  # Then, use STDCNet813M_73.91.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/STDCNet813M_73.91.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["state_dict"]

    # Removing from the checkpoint part of the head since Detectron2 is not able to match such keys and trows
    # an error.
    obj_fixed = obj.copy()
    for key in obj.keys():
        if key.startswith('fc.') or key.startswith('bn.') or key.startswith('linear.'):
            obj_fixed.pop(key)

    res = {"model": obj_fixed, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
