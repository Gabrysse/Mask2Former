import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from detectron2.evaluation import COCOPanopticEvaluator
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

logger = logging.getLogger("detectron2.evaluation.panoptic_evaluation")

class PanopticEvaluator(COCOPanopticEvaluator):
    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        super().__init__(dataset_name, output_dir)

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res, json_data)

        return results


def id2name(json_data, id):
    for c in json_data['categories']:
        if c['id'] == id:
            return c['name']

    raise KeyError

def _print_panoptic_results(pq_res, json_data):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("General Panoptic Evaluation Results:\n" + table)

    # id2name = {}
    # for c in gt_json['categories']:
    #     id2name[c['id']] = c['name']
    headers = ["category", "PQ", "SQ", "RQ"]
    data = []
    for category, result in pq_res["per_class"].items():
        try:
            name = id2name(json_data, category)
            row = [name] + [result[k] * 100 for k in ["pq", "sq", "rq"]]
        except KeyError:
            row = [category] + [result[k] * 100 for k in ["pq", "sq", "rq"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="left", numalign="center"
    )
    logger.info("Per-class Panoptic Evaluation Results:\n" + table)
