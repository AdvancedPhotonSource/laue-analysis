from pathlib import Path

from laueanalysis.indexing.index import _parse_peaks_output, _parse_npatterns_found, _parse_indexing_output


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


def test_parse_npatterns_found_from_header(tmp_path):
    """
    Validate _parse_npatterns_found when the $NpatternsFound tag is present.
    Header tag should take precedence over any $pattern sections.
    """
    content = """$filetype\tIndexFile
$NpatternsFound\t2\t// number of patterns found
$Nindexed\t12
"""
    index_file = tmp_path / "index_with_header.txt"
    index_file.write_text(content)
    assert _parse_npatterns_found(str(index_file)) == 2


def test_parse_npatterns_found_without_header_returns_zero(tmp_path):
    """
    Without $NpatternsFound header, parser should return 0 even if $pattern sections appear.
    """
    content = """$filetype IndexFile
// Found 2 patterns, they are:
$pattern0
$EulerAngles0 {0.0,0.0,0.0}
$pattern1
$EulerAngles1 {1.0,1.0,1.0}
"""
    index_file = tmp_path / "index_without_header.txt"
    index_file.write_text(content)
    assert _parse_npatterns_found(str(index_file)) == 0


def test_parse_npatterns_found_zero_patterns(tmp_path):
    """
    When file reports zero patterns explicitly, parser should return 0.
    """
    content = """$filetype IndexFile
$NpatternsFound 0
$Nindexed 0
"""
    index_file = tmp_path / "index_zero_patterns.txt"
    index_file.write_text(content)
    assert _parse_npatterns_found(str(index_file)) == 0


def test_parse_indexing_output_prefers_Nindexed_tag(tmp_path):
    """
    _parse_indexing_output should prefer the $Nindexed tag when available.
    """
    content = """$filetype IndexFile
$Nindexed 12
"""
    index_file = tmp_path / "index_nindexed_tag.txt"
    index_file.write_text(content)
    assert _parse_indexing_output(str(index_file)) == 12


def test_parse_indexing_output_without_tag_returns_zero(tmp_path):
    """
    Without $Nindexed header, parser should return 0 even if human-readable lines exist.
    """
    content = """some header line
12 reflections indexed in total
some footer line
"""
    index_file = tmp_path / "index_without_nindexed.txt"
    index_file.write_text(content)
    assert _parse_indexing_output(str(index_file)) == 0
