import pandas as pd
from astropy import table

import paths


_fwf = pd.read_fwf(paths.stis / 'stis_aperture_data.txt', infer_nrows=300, comment='#',
                   names='aperture_id aperture_name Xref Yref V2ref V3ref Xscale Yscale Betax Betay'.split())
aperture_data = table.Table.from_pandas(_fwf)
aperture_data.add_index('aperture_name')
mystery_trace_offset = 0

def predicted_trace_location(tag_hdu, return_pieces=False):
    hdr = tag_hdu[0].header
    aperture = hdr['propaper']
    aprows = aperture_data.loc[aperture]
    detector = hdr['detector']
    prefix = 'OV' if 'MAMA' in detector else 'OF'
    # prefix = 'OV' if re.findall(r'[A-WYZ]', aperture) else 'ON'
    apdata, = [row for row in aprows if prefix in row['aperture_id']]
    Yref = apdata['Yref']
    monthly_offset = hdr['moffset2']
    user_offset = hdr['postarg2']
    yloc =  Yref + monthly_offset + user_offset + mystery_trace_offset
    if return_pieces:
        pieces = dict(Yref=Yref, monthly_offset=monthly_offset, user_offset=user_offset)
        return yloc, pieces
    else:
        return yloc