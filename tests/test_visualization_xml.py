from pathlib import Path

import numpy as np
import pytest

from laueanalysis.indexing import Geometry
from laueanalysis.visualization import DataScope, load_visualization_xml


GEOMETRY = Path(__file__).parent / "data/geo/geoN_2022-03-29_14-15-05.xml"


def _step(frame, patterns):
    pattern_xml = "".join(
        f"""<pattern num="{rank}" rms_error="0.00{rank + 5}" goodness="{150 - 30 * rank}" Nindexed="{count}">
        <recip_lattice><astar>-10 5 -6</astar><bstar>-15 -2 -1</bstar><cstar>-2 13 8</cstar></recip_lattice>
        <hkl_s><h>{' '.join(['1'] * count)}</h><k>{' '.join(['0'] * count)}</k>
        <l>{' '.join(['0'] * count)}</l><PkIndex>{' '.join(str(i) for i in range(count))}</PkIndex></hkl_s>
        </pattern>"""
        for rank, count in enumerate(patterns)
    )
    peak_count = max(patterns, default=2)
    values = " ".join(str(i) for i in range(peak_count))
    return f"""<step><Xsample>{100 + frame}</Xsample><Ysample>200</Ysample><Zsample>300</Zsample>
    <depth>nan</depth><energy>20</energy><scanNum>{1001 + frame}</scanNum>
    <detector><inputImage>image_{frame}.h5</inputImage><detectorID>TEST-DET</detectorID>
    <Nx>2048</Nx><Ny>2048</Ny><ROI startx="0" starty="0" groupx="1" groupy="1"/>
    <peaksXY Npeaks="{peak_count}"><Xpixel>{values}</Xpixel><Ypixel>{values}</Ypixel>
    <Intens>{values}</Intens><Integral>{values}</Integral><Qx>{values}</Qx><Qy>{values}</Qy><Qz>{values}</Qz></peaksXY></detector>
    <indexing Npatterns="{len(patterns)}">{pattern_xml}<xtl><structureDesc>TestMaterial</structureDesc>
    <SpaceGroup>225</SpaceGroup><latticeParameters unit="nm">0.4 0.4 0.4 90 90 90</latticeParameters>
    <atom symbol="Ni" label="Ni001">0 0 0</atom></xtl></indexing></step>"""


@pytest.fixture
def legacy_xml(tmp_path):
    path = tmp_path / "indexing.xml"
    path.write_text(f"<AllSteps>{_step(0, (4, 3))}{_step(1, (2,))}</AllSteps>")
    return path


def test_load_visualization_xml_normalizes_legacy_data(legacy_xml):
    dataset = load_visualization_xml(legacy_xml)

    assert dataset.n_frames == 2
    assert dataset.n_patterns == 3
    assert dataset.n_assignments == 9
    assert dataset.frame_n_peaks.tolist() == [4, 2]
    assert dataset.pattern_ids() == ((0, 0),)
    assert dataset.pattern_ids(DataScope(patterns="all", min_indexed=0)) == (
        (0, 0),
        (0, 1),
        (1, 0),
    )
    assert dataset.crystal.name == "TestMaterial"
    assert dataset.crystal.space_group == 225
    assert np.isfinite(dataset.pattern_rotations).all()
    np.testing.assert_allclose(dataset.peaks["fit_x"][:2], [0, 1])
    np.testing.assert_allclose(dataset.peaks["qhat"][1], [1, 1, 1])


def test_declared_peak_count_preserves_rows_with_missing_fields(tmp_path):
    path = tmp_path / "partial.xml"
    path.write_text(
        """<AllSteps><step><detector><peaksXY Npeaks="3">
        <Xpixel>1 2</Xpixel><Ypixel>3 4 5</Ypixel>
        </peaksXY></detector></step></AllSteps>"""
    )
    dataset = load_visualization_xml(path)
    assert dataset.frame_n_peaks.tolist() == [3]
    assert len(dataset.peaks) == 3
    assert np.isnan(dataset.peaks["fit_x"][2])
    assert dataset.peaks["fit_y"][2] == 5


def test_load_visualization_xml_accepts_explicit_geometry(legacy_xml):
    geometry = Geometry(GEOMETRY)
    dataset = load_visualization_xml(legacy_xml, geometry=geometry)
    assert dataset.geometry is geometry


def test_missing_embedded_geometry_is_not_required(tmp_path):
    path = tmp_path / "minimal.xml"
    path.write_text(
        """<AllSteps><step><Xsample>0</Xsample><Ysample>1</Ysample><Zsample>2</Zsample>
        <detector><geoFile>/missing/geometry.xml</geoFile><Nx>2</Nx><Ny>3</Ny>
        <peaksXY Npeaks="1"><Xpixel>0</Xpixel><Ypixel>1</Ypixel></peaksXY></detector>
        </step></AllSteps>"""
    )
    dataset = load_visualization_xml(path)
    assert dataset.n_frames == 1
    assert dataset.geometry is None
    assert dataset.n_patterns == 0


def test_load_visualization_xml_validates_frame_ids(legacy_xml):
    with pytest.raises(ValueError, match="contain 2"):
        load_visualization_xml(legacy_xml, frame_ids=(1,))
