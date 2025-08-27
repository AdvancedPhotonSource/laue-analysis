from pathlib import Path

from laueanalysis.indexing.index import _parse_peaks_output


def test_parse_peaks_output_uses_peaklist_dims(tmp_path):
    """
    Validate that _parse_peaks_output correctly reads the $peakList dims:
    the second integer is the number of peaks (e.g., "$peakList 8 13").
    """
    content = """$filetype\t\tPixelPeakList
// header lines...
$peakList\t8 13\t\t// fitX fitY intens ...
"""
    peaks_file = tmp_path / "peaks_dims.txt"
    peaks_file.write_text(content)
    assert _parse_peaks_output(str(peaks_file)) == 13


def test_parse_peaks_output_zero_peaks_from_peaklist_dims(tmp_path):
    """
    When $peakList announces dims with zero peaks (e.g., "$peakList 8 0"),
    the parser should return 0 even if no data rows follow.
    """
    content = """$filetype\t\tPixelPeakList
// header lines...
$peakList\t8 0\t\t// fitX fitY intens ...
// no data rows since zero peaks
"""
    peaks_file = tmp_path / "peaks_zero.txt"
    peaks_file.write_text(content)
    assert _parse_peaks_output(str(peaks_file)) == 0


def test_parse_peaks_output_fallback_counts_rows_after_peakList_without_dims(tmp_path):
    """
    Fallback behavior when $peakList line does NOT include dims and $Npeaks is absent:
    it should count numeric rows after $peakList until a header/comment/blank terminator.
    """
    content = """$filetype\t\tPixelPeakList
// header lines...
$peakList\t\t// fitX fitY intens ...
  1.0 2.0 3 4 5 6 7 8
  9.0 2.0 3 4 5 6 7 8
$end
"""
    peaks_file = tmp_path / "peaks_fallback.txt"
    peaks_file.write_text(content)
    assert _parse_peaks_output(str(peaks_file)) == 2
