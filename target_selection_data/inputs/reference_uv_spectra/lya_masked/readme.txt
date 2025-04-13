The .spec files are csv's
they can be read with the astropy.table.Table class using format='ascii.ecsv'
They're just named .spec because the adhere to a format I created as part of the Spectrum class from github.com/parkus/spectralPhoton, which you could also use to read them if you feel like pulling that code.

The distance to the star is given in the csv header. All have been rescaled to 10 pc.

All spectra have been resampled onto the same wavelength grid of 0.05 Å over lines and 10 Å (with some partial bins) between lines. However, regard resolution with skepticism. Areas of the spectra are filled at times. The data sources in many cases do not resolve the Si II lines at ~1800 Å and the Mg II lines might not be fully resolved. That said, I have done my best to find resolved data.

The non-muscles spectra sometimes splice data from multiple stars of similar type. The stars are liste in the filenames. The goal is to have data everywhere with the best SNR and resolution possible without devoting my entire life to searching archives.

The F2 spectrum is a combination of, to the best of my knowledge, two F2V stars. However, a magnitude offset despite almost the same distane suggests otherwise. But honestly I cannot devote more time to finding more perfect stars, and they seem o match well in the FUV, so we're rolling with it.

The A6 incorporates beta Pic for the UV. Obvs this star has a disk. However, I see no evidence of emission lines in either star (overlpaping Mg II, Si II or beta pic FUV lines in cos), as with the calspec star. So I went ahead and used it. Also, some of the FUV data were taken during the possible hill sphere transit of beta pic b, but as this will likely not affect fluxes above a level of a factor of 2 or so, I did not worry about it.

The A0 spectrum is all consistent, yay!

The A6 spectrum is brigther than the A0 beyond 4200 Å. That doesn't seem right... But that is the calspec portion, so it's not my fault!

We could use a 7000 K star. But, like I said, I don't have more time to pour into this.

Lya and O I have been replaced with nans because the values in the source spectra do not represent what we would actually observe for a variety of reasons.

