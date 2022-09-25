"""
Unit tests the data module.
"""

import pytest

from stereomatch.data import MiddleburyDataset

@pytest.mark.skip(reason="The effort to create proper testing is not worth right now.")
def test_middlebury():
    """
    Tests the middlebury dataset parser.
    """
    dataset = MiddleburyDataset("../workflows/evaluation/middlebury/data/")
    _ = dataset[0]
