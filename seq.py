import cmds as c
import os

def hys(posfield, negfield, rate=200, pts=1000, fn=None, vent=True, write=1):
    ''' Hysteresis sequence '''
    # Check file name
    if fn is not None:
        fn = os.path.abspath(fn)
        fn = os.path.splitext(fn)[0]
        seqfn = fn + '.seq'
        datfn = fn + '.dat'
        dirname = os.path.dirname(fn)

        # create new file names if write is 1
        # write=2 overwrites
        if write == 1:
            # Check if file exists
            n = 0
            if os.path.isfile(seqfn):
                newseqfn = seqfn
                while os.path.isfile(newseqfn):
                    n += 1
                    newseqfn = ('_' + str(n).zfill(3)).join(os.path.splitext(seqfn))
                seqfn = newseqfn
                datfn = ('_' + str(n).zfill(3)).join(os.path.splitext(datfn))

        # Check if dir exists.  If not, create it.
        if not os.path.isdir(dirname):
                os.makedirs(dirname)

    cmds = []
    #TODO: find out if purged already
    if fn is not None:
        cmds.append(c.datafile(datfn, append=True))
    cmds.append(c.purgeseal())
    cmds.append(c.setfield(posfield, 200))
    cmds.append(c.touchdown())
    cmds.append(c.wait())
    cmds.append(c.MH(posfield, -abs(negfield), rate, pts))
    cmds.append(c.setfield(0, 200))
    if vent:
        cmds.append(c.ventseal())
    cmds.append(c.beep())

    if write:
        with open(seqfn,'w') as f:
            f.writelines(cmds)
        os.startfile(seqfn)

    return cmds

def hys(posfield, negfield, rate=200, pts=1000, fn=None, vent=True, write=1):
    ''' Hysteresis sequence '''
    # Check file name
    if fn is not None:
        fn = os.path.abspath(fn)
        fn = os.path.splitext(fn)[0]
        seqfn = fn + '.seq'
        datfn = fn + '.dat'
        dirname = os.path.dirname(fn)

        # create new file names if write is 1
        # write=2 overwrites
        if write == 1:
            # Check if file exists
            n = 0
            if os.path.isfile(seqfn):
                newseqfn = seqfn
                while os.path.isfile(newseqfn):
                    n += 1
                    newseqfn = ('_' + str(n).zfill(3)).join(os.path.splitext(seqfn))
                seqfn = newseqfn
                datfn = ('_' + str(n).zfill(3)).join(os.path.splitext(datfn))

        # Check if dir exists.  If not, create it.
        if not os.path.isdir(dirname):
                os.makedirs(dirname)

    cmds = []
    #TODO: find out if purged already
    if fn is not None:
        cmds.append(c.datafile(datfn, append=True))
    cmds.append(c.purgeseal())
    cmds.append(c.setfield(posfield, 200))
    cmds.append(c.touchdown())
    cmds.append(c.wait())
    cmds.append(c.MH(posfield, -abs(negfield), rate, pts))
    cmds.append(c.setfield(0, 200))
    if vent:
        cmds.append(c.ventseal())
    cmds.append(c.beep())

    if write:
        with open(seqfn,'w') as f:
            f.writelines(cmds)
        os.startfile(seqfn)

    return cmds
