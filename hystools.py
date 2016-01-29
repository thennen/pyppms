'''
Tools for working with hysteresis loops in python.

Tyler Hennen 2014
'''

import numpy as np
import heapq

def hc(H, M, method='avg', mpercent=10):
    ''' calculate Hc '''
    # split into two branches and fit Hc
    amp = np.max(np.abs(M))
    [H1, H2], [M1, M2] = split(H, M)
    mask1 = np.abs(M1) < (mpercent/100. * amp)
    mask2 = np.abs(M2) < (mpercent/100. * amp)
    Hcfit1 = np.polyfit(H1[mask1], M1[mask1], 1)
    Hcfit2 = np.polyfit(H2[mask2], M2[mask2], 1)
    Hc1 = -Hcfit1[1]/Hcfit1[0]
    Hc2 = -Hcfit2[1]/Hcfit2[0]

    if method == 'avg':
        return np.abs(Hc2 - Hc1) / 2.
    elif method == 'left':
        return np.min((Hc1, Hc2))
    elif method == 'right':
        return np.max((Hc1, Hc2))
    else:
        raise Exception('"method" can be \'left\', \'right\', or \'avg\'')


def recoil(Hlist, Mlist, startdef=0, satrange=('85%', '95%'), plot=False):
    '''
    Extract parameters from recoil data.  Input list of recoil curves, first
    being the major loop.
    startdef:
    0: minor loops start from remnant value
    1: minor loops start from initial value
    '''

    # only doing +- 0.5 point, can be extended later
    Mlevels = [-0.5, 0.5]

    # Find amp and offs from major loop 
    amp, offs = normalize(Hlist[0], Mlist[0], fitrange=satrange, fitbranch=False)
    Minterp = np.array([amp*m + offs for m in Mlevels])
    # H interp will be a list of H values corresponding to the M values.  Each
    # list element corresponding to a loop
    Hinterp = []

    Mr = []
    dH_ext = []
    # initial values of H, M for each minor loop
    Minit= []
    Hinit = []
    # Extract remnant values, initial values, dH values
    for H, M in zip(Hlist, Mlist):
        # find 10 points nearest 0, fit a parabola to find intercept.
        argmin = np.abs(H).argmin()
        slice_ = slice(argmin-5, argmin+5)
        pfit = np.polyfit(H[slice_], M[slice_], 2)
        mr = np.polyval(pfit, 0) 
        Mr.append(mr)
        # average first 10 points for starting M
        minit= np.mean(M[:10])
        Minit.append(minit)
        # Find M value where minor loop has half-returned to saturation
        if startdef == 0:
            mhalf = mr + (amp + offs - mr)/2
        elif startdef == 1:
            mhalf = minit + (amp + offs - minit)/2

        # interp for Minterp -- could behave badly in regions where M is not
        # monotonic
        h = np.interp(Minterp, M, H, left=np.NaN, right = np.NaN)
        Hinterp.append(h)

        # Calculate dH_ext by subtracting H_minorloop(mhalf) from H_majorloop(-mhalf)
        hhalf = np.interp(mhalf, M, H)
        majhhalf = np.interp(-mhalf+2*offs, Mlist[0], Hlist[0])

        dH_ext.append(hhalf - majhhalf)
        
    # Calculate SFD parameters
    Hnhalf_major = Hinterp[0][Mlevels.index(-0.5)]
    Hhalf_major = Hinterp[0][Mlevels.index(0.5)] 
    dH_int = [Hhalf_major - h[Mlevels.index(0.5)] for h in Hinterp]
    SFD = Hhalf_major - Hnhalf_major

    # Interpolate dH_int, dH_ext, to get iSFD, eSFD
    # First sort everything by Mstart value
    if startdef == 0:
        Mstart, dH_int, dH_ext, Hlist, Mlist = zip(*sorted(zip(Mr, dH_int, dH_ext, Hlist, Mlist)))
    elif startdef == 1:
        Mstart, dH_int, dH_ext, Hlist, Mlist = zip(*sorted(zip(Minit, dH_int, dH_ext, Hlist, Mlist)))
    if max(Mstart) > 0 and min(Mstart) < 0:
        iSFD = np.interp(0, Mstart, dH_int)
        eSFD = np.interp(0, Mstart, dH_ext)
        # interpolate to get entire M=0 loop
        nearz = heapq.nsmallest(2, Mstart, key=abs)
        nearz.sort()
        i_neg = Mstart.index(nearz[0])
        i_pos = Mstart.index(nearz[1])
        interplower = np.interp(Hlist[i_pos], Hlist[i_neg], Mlist[i_neg])
        H_zminor = Hlist[i_pos]
        weight = 1/(1 - nearz[0]/nearz[1])
        #M_zminor = [((1 - weight)*mp + weight*mn) for mp,mn in zip(Mlist[i_pos], interplower)]
        M_zminor = (1-weight)*Mlist[i_pos] + weight*interplower
    else:
        # if minor loop starting points to not cross M=0, fit parabola to dH vs
        # Mstart to get Mstart=0 value
        iSFD_fit = np.polyfit(Mstart, dH_int, 2)
        eSFD_fit = np.polyfit(Mstart, dH_ext, 2)
        iSFD = np.polyval(iSFD_fit, 0)
        eSFD = np.polyval(eSFD_fit, 0)
        # TODO: extrap for M=0 loop
        # for now, using closest to M=0 loop
        nearestz = min(Mstart, key=abs)
        H_zminor = Hlist[Mstart.index(nearestz)]
        M_zminor = Mlist[Mstart.index(nearestz)]


    if plot:
        from matplotlib import pyplot as plt

        # TODO: be explicit
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hold(True)
        for H, M, hinterp in zip(Hlist, Mlist, Hinterp):
            plt.plot(H, M)
            # plot Hinterp, Minterp
            plt.plot(hinterp, Minterp, 'o')
            # print parameters on graph
        try:
            # plot interpolated minor loop
            plt.plot(H_zminor, M_zminor, linewidth=3, color='black')
        except:
            pass

        plt.grid(True)
        plt.xlabel('H')
        plt.ylabel('M')

    pdict = {}
    pdict['SFD'] = SFD
    pdict['iSFD'] = iSFD
    pdict['eSFD'] = eSFD
    pdict['Mstart'] = Mstart
    pdict['H_zminor'] = H_zminor
    pdict['M_zminor'] = M_zminor
    pdict['plotpts'] = [(Hhalf_major, offs+amp/2),
                        (Hnhalf_major, offs-amp/2),
                        (Hhalf_major-iSFD, offs+amp/2)]

    return pdict


def easy_params(H, M, satrange=('75%', '95%'), mpercent=10, hpercent=5, plot=False):
    ''' calculate loop parameters by line extrapolation: hc, hn, hs '''
    # TODO: expand fit ranges if they result in poorly conditioned fits
    Ms, offset = normalize(H, M, satrange, fitbranch=True)
    # ignore offset
    offset = 0
    # split loop into two branches, without knowing which is which
    [H1, H2], [M1, M2] = split(H, M)
    maskflag = False
    while maskflag is False:
        Hcmask1 = np.abs(M1) < (mpercent/100. * Ms)
        Hcmask2 = np.abs(M2) < (mpercent/100. * Ms)
        if sum(Hcmask1) > 1 and sum(Hcmask2) > 1:
            maskflag = True
        else:
            mpercent += 1
    Hcfit1 = np.polyfit(H1[Hcmask1], M1[Hcmask1], 1)
    Hcfit2 = np.polyfit(H2[Hcmask2], M2[Hcmask2], 1)
    # Loop should start from max field
    # TODO: generalize for loop starting from any field

    # From first hys branch
    Hc1 = -Hcfit1[1]/Hcfit1[0]
    Hn1 = (Ms+offset-Hcfit1[1])/Hcfit1[0]
    Hs1 = (-Ms-offset-Hcfit1[1])/Hcfit1[0]
    # From second hys branch
    Hc2 = -Hcfit2[1]/Hcfit2[0]
    Hn2 = (-Ms-offset-Hcfit2[1])/Hcfit2[0]
    Hs2 = (Ms+offset-Hcfit2[1])/Hcfit2[0]
    # Calculate averages
    Hc = (np.max([Hc1, Hc2]) - np.min([Hc1, Hc2]))/2
    Hn = (np.max([Hn1, Hn2]) - np.min([Hn1, Hn2]))/2
    Hs = (np.max([Hs1, Hs2]) - np.min([Hs1, Hs2]))/2

    # Fit poly to points near H=0 to find Mr
    maxH = np.max(np.abs(H))
    maskflag = False
    while maskflag is False:
        Mrmask1 = np.abs(H1) < max(H)*hpercent/100
        Mrmask2 = np.abs(H2) < max(H)*hpercent/100
        if sum(Mrmask1) > 2 and sum(Mrmask2) > 2:
            maskflag = True
        else:
            hpercent += 1
    Mrfit1 = np.polyfit(H1[Mrmask1], M1[Mrmask1], 2)
    Mrfit2 = np.polyfit(H2[Mrmask2], M2[Mrmask2], 2)
    Mr1 = np.polyval(Mrfit1, 0)
    Mr2 = np.polyval(Mrfit2, 0)
    Mr = (np.max([Mr1, Mr2]) - np.min([Mr1, Mr2]))/2

    if plot:
        from matplotlib import pyplot as plt
        lenH = len(H)
        maxM = np.max(M)
        minM = np.min(M)
        # plot input hysteresis loop
        plt.plot(H, M)
        plt.hold(True)
        # TODO: be explicit
        # plot saturation lines
        plt.plot(H, [Ms+offset]*len(H))
        plt.plot(H, [-Ms-offset]*len(H))
        # plot Hc lines
        plt.plot(H, np.polyval(Hcfit1, H))
        plt.plot(H, np.polyval(Hcfit2, H))
        # plot Mr mark
        plt.plot([0,0], [Mr1, Mr2], 'o')
        # print parameters on graph

        plt.grid(True)
        plt.ylim((minM*1.1, maxM*1.1))
        plt.xlabel('H')
        plt.ylabel('M')

    pdict = {}
    pdict['Ms'] = Ms
    pdict['Mr'] = Mr
    pdict['Hn'] = Hn
    pdict['Hs'] = Hs
    pdict['Hc'] = Hc
    pdict['Hc1'] = Hc1
    pdict['Hc2'] = Hc2
    pdict['Hn1'] = Hn1
    pdict['Hn2'] = Hn2
    pdict['Hs1'] = Hs1
    pdict['Hs2'] = Hs2
    pdict['Mr1'] = Mr1
    pdict['Mr2'] = Mr2
    pdict['Hcslope1'] = Hcfit1[0]
    pdict['Hcslope2'] = Hcfit2[0]
    pdict['offset'] = offset
    
    return pdict



def slope(H, M, fitrange=['75%','95%'], method='avg', fitbranch=True):
    ''' calculate slope of loop by fitting a line to a field range '''
    
    fitrange = valid_fitrange(fitrange, H)

    dHdir = np.diff(H) < 0
    dHdir = np.append(dHdir[0], dHdir)
    # Find range of data for fits
    if fitbranch:
        # Field is high and decreasing
        fitmask1 = dHdir & (H < fitrange[3]) & (H > fitrange[2])
        # Field is low and increasing
        fitmask2 = ~dHdir & (H < fitrange[1]) & (H > fitrange[0])
    else:
        fitmask1 = (H < fitrange[3]) & (H > fitrange[2])
        fitmask2 = (H < fitrange[1]) & (H > fitrange[0])

    fit1 = fit2 = None
    if any(fitmask1):
        fit1 = np.polyfit(H[fitmask1], M[fitmask1], 1)
    if any(fitmask2):
        fit2 = np.polyfit(H[fitmask2], M[fitmask2], 1)

    if method == 'avg':
        if fit1 is not None and fit2 is not None:
            slope = (fit1[0] + fit2[0]) / 2
            offset = (fit1[1] + fit2[1]) / 2
        elif fit1 is not None:
            slope = fit1[0]
            offset = 0
        elif fit2 is not None:
            slope = fit2[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')
    elif method == 'left':
        if fit2 is not None:
            slope = fit2[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')
    elif method == 'right':
        if fit1 is not None:
            slope = fit1[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')

    return slope, offset
        
    
def valid_fitrange(fitrange, fieldarray=None):
    '''
    Try to make sense of input fitrange and return a valid fitrange list
    with len 4.  If any elements of fitrange are a percentage, fieldarray must
    be given
    '''
    # Convert any % in fitrange to field values
    fr = []
    if fieldarray is not None:
        maxfield = np.max(fieldarray)
        minfield = np.min(fieldarray)
    for val in fitrange:
        if isinstance(val, str):
            # if this errors, the string wasn't understood or fieldarray wasn't
            # passed.
            fr.append(float(val.strip('% ')) * maxfield/100.)
        else:
            fr.append(val)

    # Convert to len 4
    if len(fr) == 2:
        if fr[0] < 0:
            fr = [fr[0], fr[1], -fr[1], -fr[0]]
        else:
            fr = [-fr[1], -fr[0], fr[0], fr[1]]

    # TODO: make sure fitrange makes sense.
    assert fr[0] < fr[1] < fr[2] < fr[3] 

    return fr 


def normalize(H, M, fitrange=('75%', '100%'), fitbranch=False):
    ''' Normalize loop: subtract offset and scale based on data in fit range '''
    
    fitrange = valid_fitrange(fitrange, H)

    dHdir = np.diff(H) < 0
    dHdir = np.append(dHdir[0], dHdir)
    #
    if fitbranch:
        # Field is high and decreasing
        fitmask1 = dHdir & (H < fitrange[3]) & (H > fitrange[2])
        # Field is low and increasing
        fitmask2 = ~dHdir & (H < fitrange[1]) & (H > fitrange[0])
    else:
        fitmask1 = (H < fitrange[3]) & (H > fitrange[2])
        fitmask2 = (H < fitrange[1]) & (H > fitrange[0])

    mean1 = mean2 = None
    if any(fitmask1):
        mean1 = np.mean(M[fitmask1])
    if any(fitmask2):
        mean2 = np.mean(M[fitmask2])

    if mean1 is not None and mean2 is not None:
        amp = (mean1 - mean2) / 2
        offset = (mean1 + mean2) / 2
    elif mean1 is not None:
        amp = mean1
        offset = 0
    elif mean2 is not None:
        amp = mean2
        offset = 0
    else:
        raise Exception('No data points in fit range')

    return amp, offset

def hysarray(endpoints, step=None, n=None):
    if not (bool(n) ^ bool(step)):
        raise Exception('Must give either n or step, and not both')
    hysarrays = []
    if n:
        # number of data points given, calculate step
        totsweeped = sum(abs(np.diff(endpoints)))
        step = totsweeped / (n - 1)
    for i in range(len(endpoints)-1):
        currentfield = endpoints[i]
        nextfield = endpoints[i+1]
        direction = np.sign(nextfield - currentfield)
        hysarrays.append(np.arange(currentfield, nextfield, direction*step))
    # misses the last value.
    hysarrays.append([endpoints[-1]])
    return np.concatenate(hysarrays)

# That time I wanted to make simple things complicated
#def hysarray(minfield, maxfield, n=None, step=None, start='max', quads=4, startdir=1):
#    '''
#    Return an array of field values for a hysteresis loop.
#
#    startdir should be +1 or -1 (automatically set if start = 'max' or 'min'
#    
#    at present, field interval must include 0
#    '''
#    from collections import deque
#    minfield = -abs(minfield)
#    maxfield = abs(maxfield)
#    fieldvals = deque([0, minfield, 0, maxfield])
#    assert(quads > 0)
#    if not (bool(n) ^ bool(step)):
#        raise Exception('Must give either n or step, and not both')
#    if start == 'max' or start == maxfield:
#        start = maxfield
#        startdir = -1
#    elif start == 'min':
#        start = minfield
#    elif start == 0 or np.sign(start) == startdir:
#        fieldvals.rotate(startdir)
#
#    if start < minfield or start > maxfield:
#        raise Exception('Cannot start below minfield or above maxfield')
#
#    # List of arrays, to be concatenated at the end
#    hysquads = []
#    if n:
#        import warnings
#        raise warnings.warn('number of data points is not working correctly')
#        # number of data points given, calculate step
#        totsweeped = (abs(fieldvals[0] - start) 
#                     + sum([abs(fieldvals[i%4] - fieldvals[(i+1)%4]) for i in range(1,quads)]))
#        step = totsweeped / (n - 1)
#    direction = np.sign(fieldvals[0] - start)
#    hysquads.append(np.arange(start, fieldvals[0], direction*step))
#    for i in range(1, quads):
#        currentfield = fieldvals[0]
#        fieldvals.rotate(startdir)
#        nextfield = fieldvals[0]
#        direction = np.sign(nextfield - currentfield)
#        # TODO: cross zero unless it's an endpoint...
#        hysquads.append(np.arange(currentfield, nextfield, direction*step))
#    # misses the last value.
#    hysquads.append(np.array([fieldvals[0]]))
#
#
#    return np.concatenate(hysquads)




def interpM(H, M, newH):
    '''
    Interpolate one hysteresis loop/branch to new H values.  Conscious of
    increasing/decreasing H values. Can use newH with at most one direction
    change.
    '''
    Hsplit, Msplit = split(H, M)
    newHsplit, _ = split(newH, np.zeros(len(newH)))
#    if len(newHsplit) > 2:
#        raise Exception('newH changes direction more than once')
#    if len(Hsplit) > 2:
#        raise Exception('H changes direction more than once')
#
#
#
#    for newHseg in newHsplit:
#        newdir = (newHseg[1] - newHseg[0]) > 0
#        for Hseg in Hsplit:
#            dir = (Hseg[1] - Hseg[0]) > 0
#            if newdir == dir:
#                pass
#
#    return np.append(M1interp, M2interp)
   

def interpH(H, M, newM):
    ''' Interpolate a hysteresis loop to new M values '''
    pass


def split(H, M, shortest=2, loops=False):
    '''
    Split hysteresis loops into segments by field change direction.  Throw out
    segments shorter than 'shortest'
    '''
    # find indices for direction changes
    H = np.array(H)
    M = np.array(M)
    dHdir = np.diff(H) > 0
    dHindex = np.append([0,0], np.diff(dHdir))
    dHgroup = np.cumsum(dHindex)
    # split between direction changes, throwing out segments if they are
    # too short
    Hsplit = []
    Msplit = []
    for i in range(dHgroup[-1] + 1):
        if len(dHgroup[dHgroup == i]) >= shortest:
            Hsplit.append(H[dHgroup == i])
            Msplit.append(M[dHgroup == i])

    if loops:
        # Put segments together in groups of two to form loops
        Hloops = []
        Mloops = []
        for j in range(0, len(Hsplit), 2):
            try:
                Hloops.append(np.append(Hsplit[j], Hsplit[j+1]))
                Mloops.append(np.append(Msplit[j], Msplit[j+1]))
            except:
                # There aren't two more, but append the stray segment
                Hloops.append(Hsplit[j])
                Mloops.append(Msplit[j])
        return Hloops, Mloops

    return Hsplit, Msplit

def smooth(H, M, window=5):
    if len(H) == 0 or len(M) == 0:
        return H, M
    ''' Smooth H and M with rolling average'''
    weights = np.repeat(1.0, window)/window
    smoothH = np.convolve(H, weights, 'valid')
    smoothM = np.convolve(M, weights, 'valid')
    return smoothH, smoothM
