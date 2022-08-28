"""
Tests the pipeline classes.
"""

from pathlib import Path

from stereomatch.cost import SSD, SSDTexture, Birchfield
from stereomatch.aggregation import Semiglobal
from stereomatch.disparity_reduce import WinnerTakesAll, DynamicProgramming
from stereomatch.pipeline import Pipeline

from .viz import save_depthmap


def test_pipeline(sample_stereo_pair):
    """
    Tests all combinations of pipelines. The results will be placed on a folder named pipeline-out.
    """
    left, right = sample_stereo_pair

    output_dir = Path("pipeline-out")
    output_dir.mkdir(exist_ok=True)
    for device in ["cpu", "cuda"]:
        for cost, cost_name in [(SSD(32), "SSD"), (SSDTexture(32), "SSDTexture"),
                                (Birchfield(32), "Birchfield")]:
            for disp, disp_name in [(WinnerTakesAll(), "wta"), (DynamicProgramming(), "dyn")]:
                pipeline = Pipeline(cost, disp)
                disp_image = pipeline.estimate(
                    left.to(device), right.to(device))
                save_depthmap(
                    disp_image, output_dir / f"{cost_name}-{disp_name}")

                pipeline = Pipeline(cost, disp, aggregation=Semiglobal())
                pipeline.estimate(left.to(device), right.to(device))

                save_depthmap(
                    disp_image, output_dir / f"{cost_name}-sga-{disp_name}")


# def test_should_interchange_device():
#    pass
