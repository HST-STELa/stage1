


#%% actual checking

acq_filenames = []
usbl_tbl = dbutils.filter_observations(obs_tbl, usable=True)
for supfiles in usbl_tbl['supporting files']:
    if np.ma.is_masked(supfiles):
        continue
    for type, file in supfiles.items():
        if 'acq' in type.lower():
            acq_filenames.append(file)
acq_filenames = np.unique(acq_filenames)

for acq_name in acq_filenames:
    bad_acq = False
    def associated(sfs):
        return not np.ma.is_masked(sfs) and acq_name in list(sfs.values())
    assoc_obs_mask = [associated(sfs) for sfs in obs_tbl['supporting files']]

    acq_file, = dbutils.find_stela_files_from_hst_filenames(acq_name, data_dir)
    print(f'\n\nAcquistion file {acq_file.name} associated with:')
    obs_tbl[assoc_obs_mask]['start,science config,program,key science files'.split(',')].pprint(-1,-1)
    print('\n\n')
    if 'hst-stis' in acq_file.name:
        stages = ['coarse', 'fine', '0.2x0.2']
        stis.tastis.tastis(str(acq_file))
        h = fits.open(acq_file)

        if 'mirvis' in acq_file.name and 'PEAK' not in h[0].header['obsmode']:
            fig, axs = plt.subplots(1, 3, figsize=[7,3])
            for j, ax in enumerate(axs):
                data = h['sci', j+1].data
                ax.imshow(data)
                ax.set_title(stages[j])
            fig.suptitle(acq_file.name)
            fig.supxlabel('dispersion')
            fig.supylabel('spatial')
            fig.tight_layout()

            print('Click outside the plots to continue.')
            xy = utils.click_coords(fig)
    else:
        print('\nCOS data, no automatic eval routine\n')
        stages = ['initial', 'confirmation']
        h = fits.open(acq_file)
        if h[0].header['exptype'] == 'ACQ/SEARCH':
            raise NotImplementedError
        if h[0].header['exptype'] == 'ACQ/PEAKXD':
            print('PEAKXD acq')
            print(f'\txdisp offsets: {h[1].data['XDISP_OFFSET']}')
            print(f'\tcounts: {h[1].data['counts']}')
            print(f'\tslew: {h[0].header['ACQSLEWY']}')
        if h[0].header['exptype'] == 'ACQ/PEAKD':
            print('PEAKD acq')
            print(f'\tdisp offsets: {h[1].data['DISP_OFFSET']}')
            print(f'\tcounts: {h[1].data['counts']}')
            print(f'\tslew: {h[0].header['ACQSLEWX']}')
        if h[0].header['exptype'] == 'ACQ/IMAGE':
            fig, axs = plt.subplots(1, 2, figsize=[5,3])
            for j, ax in enumerate(axs):
                data = h['sci', j+1].data
                ax.imshow(np.cbrt(data))
                ax.set_title(stages[j])
            fig.suptitle(acq_file.name)
            fig.tight_layout()

            print('Click outside the plots to continue.')
            xy = utils.click_coords(fig)

    answer = input('Mark acq as good? (y/n)')
    if answer == 'n':
        bad_acq = True
    plt.close('all')

    if bad_acq:
        obs_tbl['usable'][assoc_obs_mask] = False
        obs_tbl['reason unusable'][assoc_obs_mask] = 'Target not acquired or other acquisition issue.'