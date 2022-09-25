"""
Workflows to evaluate the algorithms with the Middlebury dataset.
"""
from pathlib import Path
from typing import Tuple, Optional, List

from flytekit import workflow, task
from flytekit.types.directory import FlyteDirectory

from tqdm import tqdm
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from stereomideval.eval import Metric
import pandas as pd


from stereomatch.data import MiddleburyDataset
from stereomatch.numeric import next_power_of_2
from stereomatch.pipeline import Pipeline
from stereomatch.cli_common import create_pipeline


def _predict_depthmaps(dataset, pipeline: Pipeline):
    dataset_len = len(dataset)
    for idx in tqdm(range(dataset_len), total=dataset_len):
        item = dataset.get_stereo_pair(idx)
        left_img, right_img = item["left"], item["right"]

        left_img = rgb_to_grayscale(
            left_img.permute(2, 0, 1)).float().squeeze()
        right_img = rgb_to_grayscale(
            right_img.permute(2, 0, 1)).float().squeeze()

        pipeline.cost.max_disparity = next_power_of_2(item["max_disparity"])

        disparity = pipeline.estimate(left_img, right_img)

        yield idx, item["stereo_name"], disparity


@task(cache=True, cache_version="1.2")
def predict_disparites(cost_func: str, disp_func: str, aggr_func: Optional[str] = None) -> Tuple[FlyteDirectory, pd.DataFrame]:
    """
    Predict disparities.

    Args:
        cost_func: CLI-compatible cost function's name.
        disp_func: CLI-compatible disparity reduce function's name.
        aggr_func: CLI-compatible aggregation function's name.

    Returns:
        A directory with the predict depthmaps saved as int32 torch tensors.
        A dataframe containing information about the files. See `computes_metrics`.
    """
    pipeline = create_pipeline(
        cost_func, disp_func, aggr_method=aggr_func, max_disparity=32)
    dataset = MiddleburyDataset("middlebury/data", max_size=None)

    output_dir = Path("output/ssd32-wta")
    output_files = []
    dataset_idxs = []
    for idx, stereo_name, disparity in _predict_depthmaps(dataset, pipeline):
        output_filepath = output_dir / f"{idx}-{stereo_name}-depth.pth"
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(disparity, output_filepath)

        dataset_idxs.append(idx)
        output_files.append(str(output_filepath))

    return (FlyteDirectory(str(output_dir)),
            pd.DataFrame(dict(disparity_file=output_files, dataset_index=dataset_idxs)))


@task
def computes_metrics(disp_dir: FlyteDirectory, disp_files: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics:

    Args:
        disp_dir: Directory where the disparity maps where generated.
        disp_files: A dataframe with two columns. `disparity_file`: the predicted disparity map.
         `dataset_index`: The index corresponding to its dataset entry.
    Returns:
        A dataframe whose columns are results of different stereo evaluation metrics.
    """
    dataset = MiddleburyDataset("middlebury/data", max_size=None)
    rmse_accum = 0.0
    avgerr_accum = 0.0
    bpe_accum = 0.0
    for disp_img_file, dataset_idx in zip(disp_files["disparity_file"], disp_files["dataset_index"]):
        pred_disp = torch.load(disp_img_file)

        item = dataset.get_ground_truth(dataset_idx)
        gt_disp = item["gt_disparity"]

        rmse_accum += Metric.calc_rmse(gt_disp, pred_disp)
        avgerr_accum += Metric.calc_avgerr(gt_disp, pred_disp)
        bpe_accum += Metric.calc_bad_pix_error(gt_disp, pred_disp)

    count = len(dataset)
    metrics = pd.DataFrame(dict(rmse=[rmse_accum / count],
                                avgerr=[avgerr_accum / count],
                                bad_pix_error=[bpe_accum / count]))

    return metrics


@task
def join_metrics(conf_names: List[str], metrics_list: List[pd.DataFrame]):
    """
    Joins and displays metrics for different configurations.

    Args:
        conf_names: The name of each pipeline configuration that generated 
         a corresponding metric.
        metrics_list: A dataframe with different metrics.
    """
    for conf_name, metrics in zip(conf_names, metrics_list):
        metrics["Alg. pipeline"] = [conf_name]

    print(pd.concat(metrics_list).to_markdown(index=False))


@workflow
def metrics_wf():
    """
    Evaluation workflow.
    """
    metrics_list = []
    conf_names = []
    for cost_func, disp_func, aggr_func in [("ssd", "wta", None), ("ssd", "dyn", None),
                                            ("ssd", "dyn", "sgm")]:
        disparity_dir, disparity_pred_files = predict_disparites(
            cost_func=cost_func, disp_func=disp_func, aggr_func=aggr_func)
        metrics = computes_metrics(
            disp_dir=disparity_dir, disp_files=disparity_pred_files)

        conf_names.append(
            f"{cost_func}-{disp_func}-{str(aggr_func)}")
        metrics_list.append(metrics)

    join_metrics(conf_names=conf_names, metrics_list=metrics_list)
