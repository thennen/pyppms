'''
Python module for interactive PPMS data analysis.

Tyler Hennen 8/2014

TODO: color maps
'''
import numpy as np
import hystools as hys
import fnmatch
import heapq
import datetime
import os
import copy
import csv
from itertools import zip_longest as zipl
from matplotlib import pyplot as plt
import seq
import cmds
from container import group


# Here is where the module will look for data files.  They must end in .dat.
datadir = 'Z:\\'
bgfilepath = 'Z:\\straw.dat'
# default fit ranges
perpfr = ('75%', '99%')
defaultfr = ('90%', '99%')
ipfr = ('75%', '99%') #Currently does nothing

# For logging purposes
_version = 'CMRR 1.0'


class ppms(object):
    '''
    Class for importing, analyzing ppms data with the VSM option.

    __init__ imports data from files ending in .dat in pyppms.datadir
    with no argument, it imports the latest modified file.
    with a valid filepath as an argument, it imports that file.
    with any other string as an argument, it imports the latest modified file
    with that string as a filter

    Field and Moment arrays are stored in a list of numpy arrays:
    self.H = [H1, H2, ...],
    self.M = [M1, M2, ...].

    Methods may be used to work with data, plot, and write results to disk.
    '''

    def __init__(self, filepath=None, **kwargs):
        self.verbose = False
        if isinstance(filepath, str):
            if os.path.isfile(filepath):
                ppms.importdata(self, filepath, **kwargs)
            else:
                # Import the latest file containing the input string
                # (wildcards allowed)
                ppms.importdata(self, latest(filepath), **kwargs)
        else:
            # No arguments tries to import latest data file
            ppms.importdata(self, latest(), **kwargs)

    # Alternate constructor e.g. ppms.latest('SomeDirectory', 2)
    @classmethod
    def latest(cls, filter = '', n=1):
        return cls(latest(filter, n))


    def importdata(self, filepath, split=True, **kwargs):
        # Read header and locate start of data
        global bgfilepath
        header = []
        with open(filepath, 'r') as f:
            # Name the enumerator, so it can be used after breaking from loop
            fenum = enumerate(f)
            for num, line in fenum:
                if line == '[Data]\n':
                    skip = num + 2
                    break
                else:
                    header.append(line)
            #colnames = fenum.next()[1].split(',')

        # Just import Temp, H and M from col 2, 3, and 4
        try:
            T, H, M = np.genfromtxt(filepath, skip_header=skip, delimiter=',',
                                 unpack=True, usecols=(2,3,4))
            # drop nan
            nan = np.isnan(M)
            M = M[~nan]
            H = H[~nan]
            T = T[~nan]
        except:
            # Data is invalid, or empty
            self.H = []
            self.M = []
            self.T = []
            self.filepath = filepath
            print('Failed to load data from ' + filepath)
            self.log = 'Failed to load data.\n'
            return

        # Get sample area from header
        try:
            area = float(header[10].split(',')[1][:5])
        except:
            area = None
            if not filepath == bgfilepath and self.verbose:
                print('Area could not be found in file header.')

        # convert to uemu
        M = M * 1000000

        if split:
            # split h and m into segments
            Hsplit, Msplit = hys.split(H, M, shortest=20)
            # split T too, but hys.split sucks
            Hsplit, Tsplit = hys.split(H, T, shortest=20)

            # Remove first and last few points
            for i in range(len(Hsplit)):
                Hsplit[i] = Hsplit[i][2:-2]
                Msplit[i] = Msplit[i][2:-2]
                Tsplit[i] = Tsplit[i][2:-2]

            # put the segments back together in groups of two to form loops
            Hloops = []
            Mloops = []
            Tloops = []
            for j in range(0, len(Hsplit), 2):
                try:
                    Hloops.append(np.append(Hsplit[j], Hsplit[j+1]))
                    Mloops.append(np.append(Msplit[j], Msplit[j+1]))
                    Tloops.append(np.append(Tsplit[j], Tsplit[j+1]))
                except:
                    Hloops.append(Hsplit[j])
                    Mloops.append(Msplit[j])
                    Tloops.append(Tsplit[j])

            self.M = Mloops
            self.H = Hloops
            self.T = Tloops
        else:
            # Leave M, H, T unsplit, list with len 1
            self.M = [M]
            self.H = [H]
            self.T = [T]

        self.filepath = filepath
        self.area = area
        self._thickness = None
        self.header = header
        self.log = '{}: PyPPMS version {}\n'.format(_importtime, _version)
        self.log += '{}: Data imported from {}\n'.format(_now(), filepath)

        if not filepath == bgfilepath:
            print('Imported {} loops from {}'.format(len(self.H), os.path.split(filepath)[1]))


    def __sub__(self, bg):
        '''
        Subtract bg from loops, interp for H from the first operand.
        there should only be one loop in bg, to be broadcasted
        '''
        if not len(bg.H) == 1:
            raise Exception('second operand must contain only one loop.')
        result = copy.copy(self)
        result.M = []
        bgH = bg.H[0]
        bgM = bg.M[0]
        bgdHdir = np.diff(bgH) < 0
        bgdHdir = np.append(bgdHdir[0], bgdHdir)
        bgHdec = bgH[bgdHdir]
        bgMdec = bgM[bgdHdir]
        bgHinc = bgH[~bgdHdir]
        bgMinc = bgM[~bgdHdir]

        for H,M in zip(self.H, self.M):
            # split loops into increasing and decreasing part
            dHdir = np.diff(H) < 0
            dHdir = np.append(dHdir[0], dHdir)
            Hdec = H[dHdir]
            Mdec = M[dHdir]
            Hinc = H[~dHdir]
            Minc = M[~dHdir]

            # interp to subtract increasing and decreasing parts separately
            subtract_Mdec = Mdec - np.interp(Hdec[::-1], bgHdec[::-1], bgMdec[::-1])[::-1]
            subtract_Minc = Minc - np.interp(Hinc, bgHinc, bgMinc)
            # put subtracted parts back together in the original order
            if dHdir[0]:
                result.M.append(np.append(subtract_Mdec, subtract_Minc))
            else:
                result.M.append(np.append(subtract_Minc, subtract_Mdec))

        return result


    def __repr__(self):
        return 'ppms({})'.format(self.filepath)



#################### Methods which write data to a file #####################

    def write(self, loopnum=None, dir=None):
        ''' Write loop and parameters to file '''
        self.writeloops(loopnum=loopnum, dir=dir, log=False)
        self.writeparams(loopnum=loopnum, dir=dir, log=False)
        self.writepng(loopnum=loopnum, dir=dir, log=False)
        self.writelog(dir=dir)

    def writeloops(self, loopnum=None, dir=None, append='--Corr', log=True):
        ''' Write H, M arrays for loops to a file '''
        if loopnum is None:
            # if loopnum not given, make choice based on file name
            loopnummap = {'easy':2, 'hard':2, 'minor':'all'}
            lfilename = os.path.split(self.filepath)[1].lower()
            for k in loopnummap:
                if k in lfilename:
                    loopnum = loopnummap[k]
            if loopnum is None:
                # if none of the words in loopnummap are found, default to 'all'
                loopnum = 'all'

        loopind = self._loopind(loopnum)
        indir, fn = os.path.split(self.filepath)
        outdir = indir if dir is None else dir
        loopfn = os.path.splitext(fn)[0] + append + '.csv'
        looppath = os.path.join(outdir, loopfn)

        # if file exists, start appending numbers
        if os.path.isfile(looppath):
            matches = fnmatch.filter(os.listdir(outdir), '??'.join(os.path.splitext(loopfn)))
            if not any(matches):
                looppath = '_2'.join(os.path.splitext(looppath))
            else:
                n = np.max([int(p[-5]) for p in matches])
                looppath = ('_'+str(n+1)).join(os.path.splitext(looppath))

        # Output will be alternating H, M, H, M, ...
        # not straightforward because loops may have different lengths

        # filter out unwanted loops, convert to kOe
        H, M = [], []
        for i, [h, m] in enumerate(zip(self.H, self.M)):
            if i in loopind:
                H.append(h/1000)
                M.append(m)

        # Append the interpolated minor loop at the end if it exists
        if hasattr(self, 'H_zminor'):
            H.append(self.H_zminor)
            M.append(self.M_zminor)

        # interleave loops, with padding empty spaces with None
        # don't ask...
        raggedlooparray = zipl(*[x for t in zip(H, M) for x in t])

        with open(looppath, "wb") as f:
            # lines terminate with \r\n by default, change to \n
            excelmod = csv.excel()
            excelmod.lineterminator = '\n'
            writer = csv.writer(f, dialect=excelmod)
            writer.writerows(raggedlooparray)

        print('Loop(s) {} written to {}'.format(loopnum, looppath))
        self.log += '{}: Wrote loop(s) {} to disk: {}\n'.format(_now(), loopnum, looppath)

        if log: self.writelog(dir=dir)

    def writeparams(self, loopnum=None, dir=None, append='--Params', log=True):
        ''' Write parameters computed from loops to disk '''
        if loopnum is None:
            # if loopnum not given, make choice based on file name
            loopnummap = {'easy':2, 'hard':2, 'minor':'all'}
            lfilename = os.path.split(self.filepath)[1].lower()
            for k in loopnummap:
                if k in lfilename:
                    loopnum = loopnummap[k]
            if loopnum is None:
                # if none of the words in loopnummap are found, default to 'all'
                loopnum = 'all'

        loopind = self._loopind(loopnum)
        indir, fn = os.path.split(self.filepath)
        paramfn = os.path.splitext(fn)[0] + append + '.csv'
        outdir = indir if dir is None else dir
        parampath = os.path.join(outdir, paramfn)

        # if file exists, start appending numbers
        if os.path.isfile(parampath):
            matches = fnmatch.filter(os.listdir(outdir), '??'.join(os.path.splitext(paramfn)))
            if not any(matches):
                parampath = '_2'.join(os.path.splitext(parampath))
            else:
                n = np.max([int(p[-5]) for p in matches])
                parampath = ('_'+str(n+1)).join(os.path.splitext(parampath))

        # These parameters will be saved if they exist as class attributes
        # if they are lists, parameter[loopnum-1] will be saved

        # format [name, unit, unit_conversion_multiplier]
        paramstosave = [
                        ['Ms', 'emu/cc', 1],
                        ['Mr', 'emu/cc', 1],
                        ['Ms_t', 'uemu/mm^2', 1],
                        ['Mr_t', 'uemu/mm^2', 1],
                        ['mu_s', 'uemu', 1],
                        ['mu_r', 'uemu', 1],
                        ['Hc', 'kOe', 0.001],
                        ['Hn', 'kOe', 0.001],
                        ['Hs', 'kOe', 0.001],
                        ['SFD', 'kOe', 0.001],
                        ['iSFD', 'kOe', 0.001],
                        ['eSFD', 'kOe', 0.001],
                        ['thickness', 'nm', 1],
                        ['area', 'mm^2', 1]
                        ]

        # Create list of [param, unit, value[loopnum]]
        table = []
        for param, unit, mult in paramstosave:
            if hasattr(self, param):
                val = getattr(self, param)
                if val is not None:
                    try:
                        #table.append([param, unit, str(val[loopnum - 1] * mult)])
                        row = [param, unit]
                        row.extend([str(v*mult) for i,v in enumerate(val) if i in loopind])
                        table.append(row)
                    except:
                        # probably val is not iterable
                        table.append([param, unit, str(val * mult)])

        with open(parampath, 'w') as f:
            for l in table:
                f.write(','.join(l) + '\n')

        print('Params for loop(s) {} written to {}'.format(loopnum, parampath))
        self.log += '{}: Wrote loop parameters {} to disk: {}\n'.format(_now(), loopnum, parampath)
        if log: self.writelog(dir=dir)

    def writepng(self, loopnum=None, dir=None, append='--Plot', log=True):
        ''' Write png plot of data.  Not done yet. '''
        if loopnum is None:
            # if loopnum not given, make choice based on file name
            loopnummap = {'easy':2, 'hard':2, 'minor':'all'}
            lfilename = os.path.split(self.filepath)[1].lower()
            for k in loopnummap:
                if k in lfilename:
                    loopnum = loopnummap[k]
            if loopnum is None:
                # if none of the words in loopnummap are found, default to 'all'
                loopnum = 'all'

        loopind = self._loopind(loopnum)
        indir, fn = os.path.split(self.filepath)
        pngfn = os.path.splitext(fn)[0] + append + '.png'
        outdir = indir if dir is None else dir
        pngpath = os.path.join(outdir, pngfn)

        # if file exists, start appending numbers
        if os.path.isfile(pngpath):
            matches = fnmatch.filter(os.listdir(outdir), '??'.join(os.path.splitext(pngfn)))
            if not any(matches):
                pngpath = '_2'.join(os.path.splitext(pngpath))
            else:
                n = np.max([int(p[-5]) for p in matches])
                pngpath = ('_'+str(n+1)).join(os.path.splitext(pngpath))

        # TODO: Do a better plot
        plt.figure()
        self.plot(loopnum)
        plt.savefig(pngpath, bbox_inches='tight')
        plt.close()

        print('Plot for loop(s) {} written to {}'.format(loopnum, pngpath))
        self.log += '{}: Wrote plot for loop {} to disk: {}\n'.format(_now(), loopnum, pngpath)
        if log: self.writelog(dir=dir)

    def writelog(self, dir=None):
        ''' Append the contents of self.log to file. '''
        indir, fn = os.path.split(self.filepath)
        logfn = os.path.splitext(fn)[0] + '--log.txt'
        outdir = indir if dir is None else dir
        logpath = os.path.join(outdir, logfn)
        with open(logpath, 'a') as f:
            f.write('######################################################\n')
            f.write(self.log)
            f.write('\n')
        print('Log written to {}'.format(logpath))



#################### Methods which operate on data ###########################

    def removebg(self):
        global bgfilepath
        try:
            result = self - ppms(bgfilepath).smooth()
            result.log += '{}: Removed bg, source={}\n'.format(_now(), bgfilepath)
            if self.verbose:
                print('Removed BG from file {}'.format(os.path.split(bgfilepath)[1]))
        except:
            # BG removal failed, probably because bgfilepath is wrong.
            if self.verbose:
                print('Failed to remove BG, bgfilepath=' + bgfilepath)
            result = self
        return result


    def removeslope(self, fitrange=('-99%', '-85%', '85%', '99%'), method='avg', usefirst=False):
        '''
        Remove slope from data by fitting a line to a field range.
        method can be 'avg', 'left', or 'right'
        usefirst can be used to subtract the same line from all loops,
        determined by first loop
        '''
        # args to pass onto hystools.slope
        args = {'fitrange':fitrange, 'method':method}

        result = copy.copy(self)
        result.M = []

        first = True

        for H,M in zip(self.H, self.M):

            if not first and usefirst:
                #result.M.append(M - offset - slope*H)
                result.M.append(M - slope * H)
                continue

            slope, offset = hys.slope(H, M, **args)

            #result.M.append(M - offset - slope*H)
            result.M.append(M - slope * H)
            first = False

        result.log += '{}: Removed slope fitrange={}, method={}, usefirst={}\n'.format(_now(), fitrange, method, usefirst)
        if self.verbose:
            print('Removed slope fitrange={}, method={}, usefirst={}\n'.format(fitrange, method, usefirst))
        return result

    def subtractline(self, ms=150, fitrange=('98%', '100%')):
        ''' Subtract a line such that the loop end points are at Ms '''
        result = copy.copy(self)
        result.M = []
        for H,M in zip(self.H, self.M):
            # find current end points, subtract to get close
            amp, offs = hys.normalize(H, M, fitrange=('95%', '100%'), fitbranch=False)
            maxH = max(abs(H))
            M = M - (amp-ms)*H/maxH
            # Do it again to get decent result
            amp, offs = hys.normalize(H, M, fitrange=fitrange, fitbranch=False)
            result.M.append(M - (amp-ms)*H/maxH)

        return result


    def removedrift(self):
        ''' remove drift by subtracting linearly to match end points '''
        result = copy.copy(self)
        result.H = self.H
        result.M = []
        for H,M in zip(self.H, self.M):
            drift = np.mean(M[:50]) - np.mean(M[-50:])
            result.M.append(M + drift*np.arange(len(H))/len(H))

        result.log += '{}: Removed drift\n'.format(_now())
        return result


    def normalize(self, fitrange=('85%', '99%'), usefirst=False):
        '''
        Normalize loop based on data in fit range
        '''
        result = copy.copy(self)
        result.M = []

        fitrange = hys.valid_fitrange(fitrange)

        first = True

        for H,M in zip(self.H, self.M):

            if not first and usefirst:
                result.M.append((M - offset) / amp)
                continue

            amp, offset = hys.normalize(H, M, fitrange, fitbranch=False)

            result.M.append((M - offset) / amp)
            first = False

        result.log += '{}: Normalized, fitrange={}, usefirst={}\n'.format(_now(), fitrange, usefirst)
        return result


    def delete(self, loopnum):
        ''' delete loops '''
        loopind = self._loopind(loopnum)
        result = copy.copy(self)
        result.H = []
        result.M = []
        rng = range(len(self.H))
        result.H = [self.H[i] for i in rng if not i in loopind]
        result.M = [self.M[i] for i in rng if not i in loopind]

        result.log += '{}: deleted loopnum {}\n'.format(_now(), loopnum)
        return result


    def select(self, loopnum):
        ''' return ppms instance with loop subset '''
        loopind = self._loopind(loopnum)
        result = copy.copy(self)
        result.H = []
        result.M = []
        for i in loopind:
            result.H.append(self.H[i])
            result.M.append(self.M[i])

        result.log += '{}: selected loopnum {}\n'.format(_now(), loopnum)
        return result

    def __getitem__(self, slice):
        result = copy.copy(self)
        result.H = self.H[slice]
        result.M = self.M[slice]
        return result


    def smooth(self, window=None):
        ''' Smooth M with rolling average '''
        # if window not given, default 2% or 10, whichever is lower
        # this is to avoid over smoothing small datasets
        if len(self.H) == 0:
            # Do nothing if there's no data
            return self

        if window is None:
            minlength = min([len(h) for h in self.H])
            if minlength*0.02 < 10:
                window = int(minlength*0.02)
                window = max(window,1)
            else:
                window = 10
        result = copy.copy(self)
        result.H = []
        result.M = []
        for H,M in zip(self.H, self.M):
            smoothH, smoothM = hys.smooth(H, M, window=window)
            result.H.append(smoothH)
            result.M.append(smoothM)

        result.log += '{}: Smoothed with rolling average window={}\n'.format(_now(), window)
        return result


    def corr(self, fitrange=None, plot=False, ms=150):
        ''' Attempt to do the right correction based on file name '''
        global perpfr, ipfr

        lfilename = os.path.split(self.filepath)[1].lower()
        if 'minor' in lfilename:
            if fitrange is None:
                data = self.smooth().removebg().removeslope(('75%', '99%'), usefirst=True)
            else:
                data = self.smooth().removebg().removeslope(fitrange, usefirst=True)
            try:
                data = data.recoil_params(plot=plot)
            except:
                if self.verbose:
                    print('recoil_params() failed.')
        elif 'easy' in lfilename or 'perp' in lfilename:
            if fitrange is None:
                data = self.smooth().removebg().removeslope(perpfr)
            else:
                data = self.smooth().removebg().removeslope(fitrange)
            try:
                data = data.easy_params(plot=plot)
            except:
                if self.verbose:
                    print('easy_params() failed.')
        elif 'hard' in lfilename or 'iphys' in lfilename:
            # subtract line until its endpoints reach ms
            data = self.smooth().removebg().subtractline(ms)
            try:
                # TODO: hard_params()
                data = data.easy_params(mpercent=5, plot=plot)
            except:
                if self.verbose:
                    print('easy_params() failed.')
        else:
            if fitrange is None:
                data = self.smooth().removebg().removeslope(defaultfr)
            else:
                data = self.smooth().removebg().removeslope(fitrange)
        return data



#################### Plotting methods ########################################

    def plot(self, loopnum='all'):
        ''' plot loops'''
        loopind = self._loopind(loopnum)
        #plt.figure()
        ishold = plt.ishold()
        lines = []
        for i,(H,M) in enumerate(zip(self.H, self.M)):
            if i in loopind:
                # label plots by everything before '--' in filename
                label = os.path.split(self.filepath)[1].split('--')[0] + ' ({})'.format(i+1)
                #label = os.path.split(self.filepath)[1] + str(i+1)
                line = plt.plot(H, M, label=label)
                lines.extend(line)
                plt.hold(True)
        plt.grid(True)
        plt.title(os.path.split(self.filepath)[1])
        plt.xlabel('Field (Oe)')
        plt.ylabel('VSM signal ($\mu$ emu)')
        plt.hold(ishold)
        return lines


    def plotcycle(self, loopnum='all'):
        ''' plot loops one after another '''
        loopind = self._loopind(loopnum)

        # explicit iteration, because i can be controlled within the loop
        i = 0
        clength = len(loopind)
        fig = plt.figure()
        while -1 < i < clength:
            self.plot(i+1)
            plt.hold(False)
            plt.title(plt.getp(plt.gca(), 'title') + ' Loop {}'.format(i+1))
            # Pause runs GUI event loop...
            plt.show(block=False)
            #plt.pause(.1)
            i = _cycle(i, clength)


    def cplot(self, loopnum='all', fitrange=None):
        ''' c(orrected)plot: Remove BG according to filename and plot '''
        return self.corr(fitrange=fitrange).plot(loopnum)


################# Methods which calculate parameters from data ################

    def hc(self, method='avg', mpercent=10):
        ''' calculate Hc '''
        # split into increasing and decreasing loops
        Hc = []
        for H,M in zip(self.H, self.M):
            hc = hys.hc(H, M, method=method, mpercent=mpercent)
            Hc.append(hc)
        return Hc

    def recoil_params(self, plot=False):
        '''
        Calculate recoil parameters, SFD, iSFD, eSFD,... add them to instance
        as attributes
        '''
        result = copy.copy(self)
        # Split recoil into return loops as used by hystools.recoil
        Hsplit, Msplit = [], []
        for h, m in zip(self.H, self.M):
            try:
                [h1, h2], [m1, m2] = hys.split(h, m)
                if len(h2) > 0.8*len(h1):
                    # don't append loops that haven't returned at least 80%
                    # Cut out first 10 points, where drift is happening
                    Hsplit.append(h2[10:])
                    Msplit.append(m2[10:])
            except:
                # split failed, probably incomplete loop.
                pass

        rec_params = hys.recoil(Hsplit, Msplit, startdef=1)

        paramstosave = ['SFD', 'iSFD', 'eSFD', 'H_zminor', 'M_zminor']
        # map to these attribute names
        maptoattrib  = ['SFD', 'iSFD', 'eSFD', 'H_zminor', 'M_zminor']

        for p, m in zip(paramstosave, maptoattrib):
            setattr(result, m, rec_params[p])

        if self.verbose:
            paramstoprint = ['SFD', 'iSFD', 'eSFD']
            unitstoprint = ['Oe', 'Oe', 'Oe']
            for param, unit in zip(paramstoprint, unitstoprint):
                print(param.ljust(4) + ': {:.2f} {}'.format(rec_params[param], unit))

        if plot:
            # make summary plot
            #plt.figure()

            ishold = plt.ishold()
            plt.plot(self.H[0], self.M[0])
            plt.hold(True)
            for h, m in zip(Hsplit[1:], Msplit[1:]):
                plt.plot(h, m)
            plt.plot(rec_params['H_zminor'], rec_params['M_zminor'], color='black', linewidth=3)
            plt.scatter(*zip(*rec_params['plotpts']), c='g', s=30, zorder=3)
            plt.grid(True)
            plt.xlabel('Field (Oe)')
            plt.ylabel('M (uemu)')
            plt.title(os.path.split(self.filepath)[1])
            plt.hold(ishold)

        return result


    def easy_params(self, fitrange=('85%', '99%'), mpercent=10, plot=False):
        '''
        Calculate easy axis parameters, Ms, Mr, Hn, Hs, Hc...  add them to
        the instance as lists
        '''
        result = copy.copy(self)
        # hystools.easy_params will return a dict of computed parameters.  Save
        # these ones to the ppms instance as attributes.
        paramstosave = ['Ms', 'Mr', 'Hn', 'Hs', 'Hc']
        # map to these attribute names (assume M is in uemu)
        maptoattrib  = ['mu_s', 'mu_r', 'Hn', 'Hs', 'Hc']
        # initialize lists
        for p in maptoattrib:
            setattr(result, p, [])

        params = []
        for i, (H, M) in enumerate(zip(self.H, self.M), 1):
            try:
                params.append(hys.easy_params(H, M, fitrange, mpercent=mpercent))
                for p, m in zip(paramstosave, maptoattrib):
                    getattr(result, m).append(params[-1][p])
            except:
                # if parameter calculation fails for a loop, append nan
                if self.verbose:
                    print('Failed to calculate easy_params for loop ' + str(i))
                for m in maptoattrib:
                    getattr(result,m).append(np.nan)


        # Calculate more params
        if self.area is not None:
            # calc Mst, Mrt
            result.Ms_t = [m / self.area for m in result.mu_s]
            result.Mr_t = [m / self.area for m in result.mu_r]
            if self.thickness is not None:
                # calc Ms, Mr
                result.Ms = [m / self.thickness * 1000 for m in result.Ms_t]
                result.Mr = [m / self.thickness * 1000 for m in result.Mr_t]

        # Print out params if they exist
        if self.verbose:
            possibleparams = ['Ms', 'Mr', 'Ms_t', 'Mr_t', 'mu_s', 'mu_r', 'Hn', 'Hs', 'Hc']
            paramunits = ['emu/cc', 'emu/cc', 'uemu/mm^2', 'uemu/mm^2', 'uemu', 'uemu', 'Oe', 'Oe', 'Oe']
            print('Parameters calculated:')
            for attr, unit in zip(possibleparams, paramunits):
                if hasattr(result, attr):
                    print(attr.ljust(6) + ': ' + str(['%.2f' % v for v in getattr(result, attr)]) + ' ' + unit)
            print(' ')

        if plot:
            minH = min([min(h) for h in self.H])
            maxH = max([max(h) for h in self.H])
            minM = min([min(m) for m in self.M])
            maxM = max([max(m) for m in self.M])
            #fig = plt.figure()
            plt.hold(True)
            for H, M, p in zip(self.H, self.M, params):
                lines = plt.plot(H, M)
                color = lines[0].get_color()
                points = [(0, p['Mr1']),
                          (0, p['Mr2']),
                          (p['Hc1'], p['offset']),
                          (p['Hc2'], p['offset']),
                          ]
                plt.scatter(*zip(*points), s=30, zorder=3)
                lines = [[(minH, maxH), (p['offset']+p['Ms'], p['offset']+p['Ms'])],
                         [(minH, maxH), (p['offset']-p['Ms'], p['offset']-p['Ms'])],
                         [(p['Hc1'] + minM/p['Hcslope1'], p['Hc1'] + maxM/p['Hcslope1']), (minM, maxM)],
                         [(p['Hc2'] + minM/p['Hcslope2'], p['Hc2'] + maxM/p['Hcslope2']), (minM, maxM)],
                         ]
                for l in lines:
                    plt.plot(*l, ls='--', c=color)
            plt.grid(True)
            plt.xlabel('Field (Oe)')
            plt.ylabel('M (uemu)')
            plt.title(os.path.split(self.filepath)[1])
            plt.hold(False)


        result.log += '{}: Calculated easy axis parameters, fitrange={}\n'.format(_now(), fitrange)
        return result

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, val):
        ''' Set thickness and calculate Ms if area is also found '''
        self._thickness = val
        if (self.area is not None) & hasattr(self, 'mu_s'):
            self.Ms = [m / self._thickness / self.area * 1000 for m in self.mu_s]
            # probably has mu_r as well
            self.Mr = [m / self._thickness / self.area * 1000 for m in self.mu_r]
            print('Calculated self.Ms = ' + str(self.Ms))
            print('Calculated self.Mr = ' + str(self.Mr))


#################### Others #####################

    def genminorseq(self, loopnum=2):
        offset = 500
        Hc = self.hc()[loopnum-1]
        seqpath = os.path.splitext(self.filepath)[0] + 'MinorLoops.seq'
        # offset minor loops wrt Hc because drift
        minorseq(np.linspace(Hc - 900 - offset, Hc + 900 - offset, 7), seqpath, run=True)

        print('Generated minor loop sequence for {}, loop {}, Hc={}'.format(self.filepath, loopnum, Hc))

    def _loopind(self, loopnum_in):
        '''
        Makes sense of loopnum input in context of the data contained in
        the instance.  Return a list of valid loop indices. loopnum starts
        from 1, not 0
        '''
        l = len(self.H)
        if loopnum_in == 'all':
            return range(l)
        elif not np.iterable(loopnum_in):
            loopnum_in = [int(loopnum_in)]
        else:
            try:
                loopnum_in = [int(n) for n in loopnum_in]
            except:
                raise Exception('loopnum must be convertable to int')

        if all([abs(lnum) <= l for lnum in loopnum_in]) and not 0 in loopnum_in:
            loopnum_out = [n%l if n < 0 else (n-1)%l for n in loopnum_in]
        else:
            raise Exception('loopnum is out of range.  Valid range is +- 1:len(self.H)')
        return loopnum_out


def minorseq(rfields, filepath, run=True):
    ''' Create ppms vsm minor loop recipe file. '''
    begin = ('WAI WAITFOR 180 0 1 0 1 0\n'
              'VSMLS 1 0 0 0 0 0\n'
              'REM \n'
              'VSMMH 1 34078727 0 0 0 2 35 1 0 2 0 2 6 -90000 0 90000 196 1 25 50 0 2 1 0 1 0 "A/C,0,10,10,0" \n')
    end = ('FLD FIELD 0.0 196.1 0 0\n'
           'REM ')
    with open(filepath, 'w') as f:
        f.write(begin)
        cmd = 'VSMMH 1 34078727 0 0 0 2 35 1 0 2 0 2 6 {} 0 {} 196 1 25 50 0 2 1 0 1 0 "A/C,0,10,10,0" \n'
        for r in rfields:
            f.write(cmd.format(int(-abs(r)), 90000))
        f.write(end)
    if run:
        os.startfile(filepath)

def genminorseq(loopnum=2):
    ''' Generate and open minor loop sequence using latest easy axis data '''
    ppms.latest('easy').corr().genminorseq(loopnum)


def latest(filter='', n=1):
    '''
    Return the path of the latest file modified in datadir, which contains
    filter. Wildcards allowed. Case insensitive.
    n may be increased specify earlier files.
    '''
    global datadir
    if not os.path.isdir(datadir):
        raise Exception('datadir is not a dir')
    filter = '*' + filter + '*'
    # Look for most recent data file
    def ipath(path):
        for path, dir, files in os.walk(path):
            # get the whole path, so that directory may be specified in the filter
            fpath = [os.path.join(path, fname) for fname in files]
            for f in fnmatch.filter(fpath, filter):
                if os.path.splitext(f)[1].lower() == '.dat':
                    yield os.path.join(path, f)
    try:
        if n > 1:
            nlargest = heapq.nlargest(n, ipath(datadir), key=os.path.getmtime)
            return nlargest[n-1]
        else:
            return max(ipath(datadir), key=os.path.getmtime)
    except:
        raise Exception('Fewer than {} filenames match the specified filter'.format(n))


def nlatest(filter='', n=1):
    '''
    Same as latest except it returns the whole list of the n latest modified files
    '''
    global datadir
    if not os.path.isdir(datadir):
        raise Exception('datadir is not a dir')
    filter = '*' + filter + '*'
    # Look for most recent data file
    def ipath(path):
        for path, dir, files in os.walk(path):
            # get the whole path, so that directory may be specified in the filter
            fpath = [os.path.join(path, fname) for fname in files]
            for f in fnmatch.filter(fpath, filter):
                if os.path.splitext(f)[1].lower() == '.dat':
                    yield os.path.join(path, f)
    try:
        nlargest = heapq.nlargest(n, ipath(datadir), key=os.path.getmtime)
        return nlargest
    except:
        raise Exception('Fewer than {} filenames match the specified filter'.format(n))


def latestdir(dirfilter='', n=1):
    '''
    Return the path of directory containing the latest file modified in datadir.
    Wildcards allowed. Case insensitive.
    '''
    global datadir
    if not os.path.isdir(datadir):
        raise Exception('datadir is not a dir')
    dirfilter = '*' + dirfilter + '*'
    # Look for most recent data file
    def ipath(path):
        for path, dir, files in os.walk(path):
            if fnmatch.fnmatch(path, dirfilter):
                for f in files:
                    if os.path.splitext(f)[1].lower() == '.dat':
                        yield os.path.join(path, f)
    try:
        if n > 1:
            nlargest = heapq.nlargest(n, ipath(datadir), key=os.path.getmtime)
            nlargestdir = [os.path.split(p)[0] for p in nlargest]
            return nlargestdir[n-1]
        else:
            largest = max(ipath(datadir), key=os.path.getmtime)
            return os.path.split(largest)[0]
    except:
        raise Exception('Fewer than {} filenames match the specified filter'.format(n))


def plotlatestcycle(filter=''):
    ''' Plot cycle through the loops in the latest data file '''
    ppms(latest(filter)).corr().plotcycle()


def plotlatest(filter='', loopnum='all'):
    ''' Correct and plot the latest data written to disk '''
    ppms(latest(filter)).corr().plot(loopnum=loopnum)


def plotcycle(filter='', n=10, loopnum='all'):
    ''' plot data files one at a time '''
    files = nlatest(filter, n)
    for i,f in enumerate(files):
        print(str(i+1) + ': ' + f)
    data = [ppms(f) for f in files]
    dlength = len(data)
    i = 0
    while -1 < i < dlength:
        data[i].cplot(loopnum=loopnum)
        plt.hold(False)
        # Pause runs GUI event loop...
        plt.pause(.1)
        i = _cycle(i, dlength)


def nplot(filter = '', n=10, loopnum='all', legend=True):
    ''' Plot from n files at once '''
    washeld = plt.ishold()
    if not washeld:
        # clear current axis if not hold
        plt.cla()

    #fig, ax = plt.subplots()
    lines = []
    for fn in reversed(nlatest(filter, n)):
        # plot oldest first
        line = ppms(fn).cplot(loopnum)
        lines.extend(line)
        plt.hold(True)
    plt.hold(False)
    if legend:
        plt.legend(loc='best')

    # leave plot in previous hold state
    if not washeld:
        plt.hold(False)

    return lines


def live(loopnum='all', corr=False, fitrange=None, hold=False, legend=True):
    ''' Enter infinite loop that keeps plotting from the most recent data '''
    # passing hold=True will keep plots even when file changes
    path = latest()
    print('!Ctrl-C will set you free!')

    washeld = plt.ishold()
    if not washeld:
        # clear current axis if not hold
        plt.cla()

    ccycle = plt.rcParams['axes.color_cycle']
    if corr:
        data = ppms(path).corr(fitrange=fitrange)
    else:
        data = ppms(path)
    lines = data.plot(loopnum=loopnum)
    # give inital xlim and ylim
    try:
        plt.autoscale()
        maxH = np.max(np.abs(data.H[0]))
        maxM = np.max(np.abs(data.M[0]))
        xlimit = plt.xlim()
        ylimit = plt.ylim()
        # Change limit to +-max in data if that's bigger than autorange
        if -maxH < xlimit[0] or maxH > xlimit[1] or -maxM < ylimit[0] or maxM > ylimit[1]:
            plt.xlim((-maxH * 1.1, maxH * 1.1))
            plt.ylim((-maxM * 1.1, maxM * 1.1))
    except:
        pass
    timer = 1
    while True:
        data = ppms(path)
        title = '!' + os.path.split(path)[1] + '!'
        # get colors of last lines
        colors = [l.get_color() for l in lines]
        # clear the last lines
        for l in lines:
            l.remove()
        plt.hold(True)
        # replace lines with the updated ones
        if corr:
            lines = data.cplot(loopnum=loopnum)
        else:
            lines = data.plot(loopnum=loopnum)
        # making sure lines don't change color, and additional lines get the
        # right color ...
        for l, c in zip(lines, colors):
            l.set_color(c)
        if len(lines) > len(colors):
            try:
                nextcolori = (ccycle.index(c[-1]) + 1) % len(ccycle)
                nextcolor = ccycle[nextcolori]
                lines[-1].set_color(nextcolor)
            except:
                # give up
                pass

        if legend:
            plt.legend(loc='best')

        # leave plot in previous hold state during pause
        if not washeld:
            plt.hold(false)

        # display cute blinking ! ! around title
        if timer % 2:
            plt.title(title.replace('!',''))
        else:
            plt.title(title)

        if not timer % 20:
            # check again for latest filename
            oldpath = path
            path = latest()
            if not (path == oldpath) and hold:
                # redefine lines so the existing ones don't get clobbered
                plt.hold(True)
                lines = plt.plot()
        timer += 1
        plt.pause(2)


def wiz(folderfilter='', filefilter=''):
    ''' Run through automated analysis of all ppms data files in a directory (non-recursive) '''
    # This is NOT FINISHED
    # use latest directory by default
    dir = latestdir(folderfilter, 1)
    print('PPMS analysis wizard: ' + dir)
    print('Enter q at any prompt to quit.')
    files = fnmatch.filter(os.listdir(dir), '*' + filefilter + '*.dat')
    recognized = []
    perp_files = fnmatch.filter(files, '*perp*')
    recognized.extend(perp_files)
    ip_files = fnmatch.filter(files, '*ip*')
    recognized.extend(ip_files)
    recoil_files = fnmatch.filter(files,'*minor*')
    recognized.extend(recoil_files)
    notrecognized = [f for f in files if f not in recognized]
    if any(notrecognized):
        print('Did not recognize the following data files:')
        for nr in notrecognized:
            print(os.path.split(nr)[1])
    # extract sample names.  should have -- separator.  Might still work otherwise..
    sample_names = [f.split('--')[0].split('-Easy')[0].split('-Hard')[0].split('-Minor')[0] for f in recognized]
    sample_names = sorted(list(set(sample_names)))

    # Remove filename ambiguities e.g. sample1_EasyAxis.dat, sample1_EasyAxis2.dat
    for sn in sample_names:
        ambig_perp = fnmatch.filter(perp_files, sn + '*')
        ambig_ip = fnmatch.filter(ip_files, sn + '*')
        ambig_recoil = fnmatch.filter(recoil_files, sn + '*')
        if len(ambig_perp) > 1:
                print('Found ambiguity for {} perp axis data.'.format(sn))
                for i, fn in enumerate(ambig_perp, 1):
                    print(str(i) + ': ' + os.path.split(fn)[1])
                while True:
                    choice = raw_input('Which data file should be used? ')
                    if choice == 'q':
                        return
                    try:
                        choice = int(choice)
                    except:
                        continue
                    if 1 <= choice <= len(ambig_perp):
                        ambig_perp.remove(ambig_perp[choice-1])
                        perp_files = [ef for ef in perp_files if ef not in ambig_perp]
                        break
                    else:
                        print('not a valid choice')
        if len(ambig_ip) > 1:
                print('Found ambiguity for {} ip axis data.'.format(sn))
                for i, fn in enumerate(ambig_ip, 1):
                    print(str(i) + ': ' + os.path.split(fn)[1])
                while True:
                    choice = raw_input('Which data file should be used? ')
                    if choice == 'q':
                        return
                    try:
                        choice = int(choice)
                    except:
                        continue
                    if 1 <= choice <= len(ambig_ip):
                        ambig_ip.remove(ambig_ip[choice-1])
                        ip_files = [ef for ef in ip_files if ef not in ambig_ip]
                        break
                    else:
                        print('not a valid choice')
        if len(ambig_recoil) > 1:
                print('Found ambiguity for {} recoil data.'.format(sn))
                for i, fn in enumerate(ambig_recoil, 1):
                    print(str(i) + ': ' + os.path.split(fn)[1])
                while True:
                    choice = raw_input('Which data file should be used? ')
                    if choice == 'q':
                        return
                    try:
                        choice = int(choice)
                    except:
                        continue
                    if 1 <= choice <= len(ambig_recoil):
                        ambig_recoil.remove(ambig_recoil[choice-1])
                        recoil_files = [ef for ef in recoil_files if ef not in ambig_recoil]
                        break
                    else:
                        print('not a valid choice')

    nsamples = len(sample_names)
    if nsamples == 0:
        print('No data files.')
        return
    print('Analyzing the following data files:')
    hasperp = [False]*nsamples
    hasip = [False]*nsamples
    hasrecoil = [False]*nsamples
    for i, sn in enumerate(sample_names):
        for ef in perp_files:
            if ef.startswith(sn):
                print(os.path.split(ef)[1])
                hasperp[i] = ef
                break
        for hf in ip_files:
            if hf.startswith(sn):
                print(os.path.split(hf)[1])
                hasip[i] = hf
                break
        for rf in recoil_files:
            if rf.startswith(sn):
                print(os.path.split(rf)[1])
                hasrecoil[i] = rf
                break

    choice = raw_input('Continue? ')
    if choice == 'q' or choice == 'n':
        return

    # Try to get thicknesses for perp axis loops
    thickness = [False]*len(ef)
    for i, ef in enumerate(perp_files):
        ef = os.path.split(ef)[1]
        while True:
            choice = raw_input('Thickness for {}, (s)kip, (s!)kip all, (q)uit? '.format(ef))
            if choice == 's!':
                break
            elif choice =='s':
                break
            elif choice == 'q':
                return
            try:
                thickness[i] = float(choice)
                break
            except:
                print('Invalid entry.')
        if choice == 's!':
            break

    # save ppms instances to list and return at the end
    returnlist = []
    for sn, hp, hip, hr in zip(sample_names, hasperp, hasip, hasrecoil):
        if hp:
            # import the perp axis loop for this sample
            p = ppms(os.path.join(dir, hp))
            # do the standard correction
            fig = plt.figure()
            pcorr = p.corr(plot=True)
            t = thickness[perp_files.index(hp)]
            if t:
                pcorr.thickness = t
            returnlist.append(pcorr)
            #pcorr.write(loopnum=1)
            if hip:
                ip = ppms(os.path.join(dir, hip))
                plt.hold()
                ipcorr = ip.corr(ms=pcorr.mu_s[0], plot=True)
                returnlist.append(ipcorr)
                #ipcorr.write(loopnum=1)
        elif hip:
            # Hard axis but no perp axis
            ip = ppms(os.path.join(dir, hip))
            plt.figure()
            ipcorr = ip.corr(plot=True)
            #ipcorr.write(loopnum=1)

        if hr:
            r = ppms(os.path.join(dir, hr))
            plt.figure()
            rcorr = r.corr(plot=True)
            returnlist.append(rcorr)
            #rcorr.write()

    return returnlist

    # TODO: show plots of analysis and ask if it went okay
    # TODO: allow changing some parameters on the fly, including loopnum
    # TODO: write all results to aggregrate file, and individual files


def datadrive(driveletter):
    ''' change the datadir and bgfilepath drive letter '''
    global datadir
    global bgfilepath
    datadir = driveletter + datadir[1:]
    bgfilepath = driveletter + bgfilepath[1:]


def _now():
    ''' get the current time string for logging '''
    return str(datetime.datetime.now())


def _cycle(i, length):
    ''' handle the iteration variable for plot cycle'''
    rinput = raw_input('plot ' + str(i+1) + ': #/(n)ext/(b)ack/(q)uit? ')
    if rinput == '':
        rinput = 'n'
    options = {
               'n': i + 1,
               'b': i - 1,
               'q': -1
               }
    if rinput.lower()[0] in options.keys():
        return options[rinput.lower()[0]]
    elif 0 < int(rinput) < length + 1:
        return int(rinput) - 1
    else:
        return i+1

# some bindings for the exceptionally lazy
pl = plotlatest
plc = plotlatestcycle
gms = genminorseq
pc = plotcycle

# for logging when the module was imported
_importtime = _now()
