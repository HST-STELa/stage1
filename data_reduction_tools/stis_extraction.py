from pathlib import Path

import pandas as pd
from astropy import table

import paths

script_dir = Path(__file__).resolve().parent

_fwf = pd.read_fwf(script_dir / 'stis_aperture_data.txt', infer_nrows=300, comment='#',
                   names='aperture_id aperture_name Xref Yref V2ref V3ref Xscale Yscale Betax Betay'.split())
aperture_data = table.Table.from_pandas(_fwf)
aperture_data.add_index('aperture_name')
mystery_trace_offset = 0


def predicted_trace_location(tag_hdu):
    hdr = tag_hdu[0].header
    aperture = hdr['aperture']
    aprows = aperture_data.loc[aperture]
    detector = hdr['detector']
    prefix = 'OV' if 'MAMA' in detector else 'OF'
    apdata, = [row for row in aprows if prefix in row['aperture_id']]
    Yref = apdata['Yref']
    offset = hdr['moffset2']
    return Yref + offset + mystery_trace_offset