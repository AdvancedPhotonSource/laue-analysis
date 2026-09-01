# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Packaging declarations for the private simulation resources."""

from importlib import resources

import lauelab.analysis._vendor.jzt as jzt


def test_private_jzt_resources_are_available():
    package = resources.files(jzt)

    assert package.joinpath("elementData.xml").is_file()
    assert package.joinpath("README.md").is_file()
    assert not package.joinpath("LauePattern_allspots.py").is_file()
