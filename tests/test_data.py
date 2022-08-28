"""
Unit tests the data module.
"""
from stereomatch.data import MiddleburyDataset

def test_middlebury():
    """
    Tests the middlebury dataset parser.
    """
    dataset = MiddleburyDataset("../workflows/evaluation/middlebury/data/")
    item = dataset[0]

    __import__("ipdb").set_trace()

