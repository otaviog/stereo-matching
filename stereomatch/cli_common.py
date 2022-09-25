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


def create_pipeline(cost_method: str, disp_method: str, aggr_method: str,
                    max_disparity: int = 32) -> Pipeline:
    """
    Creates a pipelines using strings to refer to function/methods names.

    Available cost functions are:
    >>> list(COST_METHODS.keys())

    Available disparity methods are:
    >>> list(DISPARITY_METHODS.keys())

    Available aggregation methods are:
    >>> list(AGGREGATION_METHODS.keys())

    Args:
        cost_method: Refer to the available functions.
        disp_method: Refer to the available methods.
        aggr_method: Refer to the available methods.
        max_disparity: Maximum disparity that the cost function should consider.
    """

    aggregation_method = AGGREGATION_METHODS.get(aggr_method, None)
    if aggregation_method is not None:
        aggregation_method = aggregation_method()

    return Pipeline(COST_METHODS[cost_method](max_disparity),
                    DISPARITY_METHODS[disp_method](),
                    aggregation=aggregation_method)
