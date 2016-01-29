from multivu import seq
import os
import numpy as np

# Create sequence for forc curves on QD VSM attachment

# Sequence and data files written to datadir:
datadir = r'C:\DATA\forc'
# Create a subdirectory for sequence and datafiles
subdirprefix = 'forc'
dirlist = [f for f in os.listdir(datadir) if f.startswith(subdirprefix)]
existing = [0]
for d in dirlist:
    try:
        existing.append(int(d.split('_')[-1]))
    except:
        pass
fnum = max(existing) + 1
subdir = '{}_{:03d}'.format(subdirprefix, fnum)
fulldir = os.path.join(datadir, subdir)
if not os.path.isdir(fulldir):
    os.makedirs(fulldir)

################ FORC recipe #####################

maxfield = 10000
rfields = np.linspace(0, -10000, 250)
# Sweep rate (Oe/s)
rate = 40
# Data points per branch
pts = 500

forc = seq()

forc.setfield(maxfield, rate)
forc.MH_loop(maxfield, -maxfield, rate, pts)
for i,Hr in enumerate(rfields):
    fname = 'forc_data_{}.dat'.format(i)
    fpath = os.path.join(fulldir, fname)
    # Create a new file
    forc.datafile(fpath)
    forc.setfield(Hr, rate)
    forc.MH(Hr, maxfield, rate, pts)

##################################################

# Write sequence to disk
seq_fpath = os.path.join(fulldir, 'forc_{:03d}.seq'.format(fnum))
forc.write(seq_fpath)
# Open it
forc.open()
