#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from laueanalysis.indexing import index, IndexingResult


PEAKS_TEMPLATE_WITH_DEPTH = """$filetype\t\tPixelPeakList
$inputImage\t\t{input_image}
$xdim\t\t2048
$ydim\t\t2048
$xDimDet\t\t2048
$yDimDet\t\t2048
$startx\t\t0
$endx\t\t2047
$groupx\t\t1
$starty\t\t0
$endy\t\t2047
$groupy\t\t1
$exposure\t\t1
$CCDshutterIN\t1
$Xsample\t\t2502
$Ysample\t\t-4029.29
$Zsample\t\t-9262.29
$depth\t\t40\t\t// depth for depth resolved images (micron)
$scanNum\t\t274036
$beamBad\t\t0
$lightOn\t\t0
$energy\t\t19.9999
$hutchTemperature\t24.6111
$sampleDistance\t0
$monoMode\t\twhite slitted
$dateExposed\t2022-04-20T07:59:00-06:00
$detector_ID\tPE1621 723-3335
//
// fitted peak positions relative to the start of the ROI (not detector origin)
//    peak positions are in zero based binned pixels
$Npeaks\t\t1
$peakList\t8 1\t\t// fitX fitY intens integral hwhmX hwhmY tilt chisq
      643.081     1428.503       1806.1321        38.63823      1.578      1.733   178.3092   0.059983
"""


PEAKS_TEMPLATE_NO_DEPTH = """$filetype\t\tPixelPeakList
$inputImage\t\t{input_image}
$xdim\t\t2048
$ydim\t\t2048
$xDimDet\t\t2048
$yDimDet\t\t2048
$startx\t\t0
$endx\t\t2047
$groupx\t\t1
$starty\t\t0
$endy\t\t2047
$groupy\t\t1
$exposure\t\t1
$CCDshutterIN\t1
$Xsample\t\t2502
$Ysample\t\t-4029.29
$Zsample\t\t-9262.29
$scanNum\t\t274036
$beamBad\t\t0
$lightOn\t\t0
$energy\t\t19.9999
$hutchTemperature\t24.6111
$sampleDistance\t0
$monoMode\t\twhite slitted
$dateExposed\t2022-04-20T07:59:00-06:00
$detector_ID\tPE1621 723-3335
//
// fitted peak positions relative to the start of the ROI (not detector origin)
//    peak positions are in zero based binned pixels
$Npeaks\t\t1
$peakList\t8 1\t\t// fitX fitY intens integral hwhmX hwhmY tilt chisq
      643.081     1428.503       1806.1321        38.63823      1.578      1.733   178.3092   0.059983
"""


def _make_fake_run_peaksearch(template: str):
    def _fake_run_peaksearch(input_image: str, output_dir: str, *args, **kwargs):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(input_image).stem
        peaks_path = out_dir / f"peaks_{base_name}.txt"
        content = template.format(input_image=input_image)
        peaks_path.write_text(content)
        # Return code tuple: success, stdout, stderr, return_code
        return True, "fake peaksearch ok", "", 0
    return _fake_run_peaksearch


def _fake_run_p2q(*args, **kwargs):
    # Do nothing, just return success. We won't create a p2q output file.
    return True, "fake p2q ok", "", 0


@pytest.mark.parametrize("template_has_depth", [True, False])
def test_depth_override_applied_in_peaks_header(template_has_depth: bool):
    # Use real file paths for consistency with index() naming, but processing is mocked.
    test_file = os.path.join("tests", "data", "gdata", "Al30_thick_wire_55_50.h5")
    geo_file = os.path.join("tests", "data", "geo", "geoN_2022-03-29_14-15-05.xml")
    crystal_file = os.path.join("tests", "data", "crystal", "Al.xtal")

    assert os.path.exists(test_file), "Missing test input image"
    assert os.path.exists(geo_file), "Missing geometry file"
    assert os.path.exists(crystal_file), "Missing crystal file"

    override_value = 123.45

    template = PEAKS_TEMPLATE_WITH_DEPTH if template_has_depth else PEAKS_TEMPLATE_NO_DEPTH

    with tempfile.TemporaryDirectory() as temp_dir, \
         patch("laueanalysis.indexing.index._run_peaksearch", _make_fake_run_peaksearch(template)), \
         patch("laueanalysis.indexing.index._run_p2q", _fake_run_p2q):
        # Run indexing with depth_override; executables are mocked
        result = index(
            input_image=test_file,
            output_dir=temp_dir,
            geo_file=geo_file,
            crystal_file=crystal_file,
            depth_override=override_value
        )

        # Basic assertions
        assert isinstance(result, IndexingResult)
        assert result.success is True
        assert "peaks" in result.output_files
        peaks_path = result.output_files["peaks"]
        assert os.path.exists(peaks_path)

        # The peaks header should contain the overridden $depth value
        content = Path(peaks_path).read_text()
        # Look for the $depth line and extract the numeric value
        depth_line = next((l for l in content.splitlines() if l.strip().startswith("$depth")), None)
        assert depth_line is not None, "Depth line was not inserted into peaks header"
        # Ensure the override is present as a literal string
        assert f"$depth\t\t{override_value}" in depth_line

        # Command history should mention the override
        assert any("override_depth" in cmd for cmd in result.command_history), \
            "override_depth entry not found in command_history"

        # Optional: log mentions the applied override
        assert "Depth override applied" in result.log
