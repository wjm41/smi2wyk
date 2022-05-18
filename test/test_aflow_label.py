import pytest
from ccdc import io

from smi2wyk.wren_code.utils import get_aflow_label_with_aflow_from_ccdc_crystal, return_spacegroup_number, CifStringParser, get_aflow_label_spglib


@pytest.fixture()
def csd_reader():
    return io.EntryReader()

@pytest.mark.parametrize(
    "entry_name",
    [
        'ABEHOE',
        'ABOJUX',
        'ABOVUJ',
        'ABUWAY',
        'AMAPIP',
        pytest.param('ABIMUW', marks=pytest.mark.xfail(reason='unknown b0rk')),
    ],
)
def test_aflow_label_from_aflow(csd_reader, entry_name):

    entry = csd_reader.entry(entry_name)
    crystal = entry.crystal
    
    wyckoff_label_aflow = get_aflow_label_with_aflow_from_ccdc_crystal(crystal)
    spg_aflow = return_spacegroup_number(wyckoff_label_aflow)
    spg_ccdc = crystal.spacegroup_number_and_setting[0]

    assert (spg_ccdc == spg_aflow)

@pytest.mark.parametrize(
    "entry_name",
    [
        'ABIMUW',
    ],
)    
def test_aflow_label_from_spglib(csd_reader, entry_name):

    entry = csd_reader.entry(entry_name)
    crystal = entry.crystal
    
    cif_string = crystal.to_string(format='cif')
    struct = CifStringParser(cif_string, occupancy_tolerance=10).get_structures()[0]
    wyckoff_label_spglib = get_aflow_label_spglib(struct)
    spg_spglib = return_spacegroup_number(wyckoff_label_spglib)
    spg_ccdc = crystal.spacegroup_number_and_setting[0]

    assert (spg_ccdc == spg_spglib)