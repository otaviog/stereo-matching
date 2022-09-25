"""
Common utilities for CLI applications.
"""
from .cost import SSD, SSDTexture, Birchfield
from .aggregation import Semiglobal
from .disparity_reduce import WinnerTakesAll, DynamicProgramming
from .pipeline import Pipeline

COST_METHODS = {
    "ssd": SSD,
    "ssd-texture": SSDTexture,
    "birchfield": Birchfield
}

AGGREGATION_METHODS = {
    "sgm": Semiglobal,
}

DISPARITY_METHODS = {
    "wta": WinnerTakesAll,
    "dyn": DynamicProgramming
}


def create_pipeline(cost_method: str, disp_method: str, aggr_method: str, max_disparity: int = 32) -> Pipeline:
    aggregation_method = AGGREGATION_METHODS.get(aggr_method, None)
    if aggregation_method is not None:
        aggregation_method = aggregation_method()

    return Pipeline(COST_METHODS[cost_method](max_disparity),
                    DISPARITY_METHODS[disp_method](),
                    aggregation=aggregation_method)
