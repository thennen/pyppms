'''
Python module for interactive PPMS data collection/analysis.

Tyler Hennen 2/2015
'''
import numpy as np
import os
import fnmatch
import time
import seq
from matplotlib import pyplot as plt
import pyppms
import pickle

# Here is where the module will look for data files.  They must end in .dat.
#datadir = pyppms.datadir
#datadir = 'Z:\\'
datadir = 'C:\\Users\\Versalab\\Documents\\THenn'
# TODO: fix this

# For logging purposes
_version = 'CMRR 1.0'

class group(object):
    '''
    Container for ppms objects, representing an experiment.

    Can import data from file, or collect it

    Sample parameters can be stored for easy plotting, reporting

    This will be pretty cool.

    Getting the basics in first, can add sophistication

    TODO: log all activity
    '''

    def __init__(self, name=None, snames=None, params=None, dir=None):
        if name is None:
            self.name = ''
        else:
            self.name = name
        if snames is None:
            self.snames = []
        else:
            self.snames = snames
        # dict of parameters
        if params is None:
            self.params = {}
        else:
            self.params = params
        # list of ppms instances
        self.data = [None] * len(self.snames)

        # Pick a dir to store files
        if dir is None:
            if len(self.name) > 5 and all([s.isdigit() for s in self.name[:6]]):
                self.dir = self.name
            else:
                self.dir = time.strftime('%d%m%y') + '-' + self.name
                # whatever.
            self.dir = os.path.join(datadir, self.dir)
            if os.path.isdir(self.dir):
                print(self.dir + ' already exists.  Using anyway.')
        else:
            self.dir = datadir

    def __getitem__(self, key):
        # TODO: allow string indexing
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.snames[key]
        del self.data[key]
        for k in self.params.keys():
            del self.params[k][key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        out = [] 
        out.append('PPMS container')
        out.append('Name: ' + self.name)
        out.append('Samples:')
        for i,sn in enumerate(self.snames):
            out.append('{}:\t{}'.format(i, sn))
            if i < len(self.data):
                out.append('\t{}'.format(self.data[i]))
            else:
                out.append('\tNo Data')
            for key in self.params.keys():
                out.append('\t{} = {}'.format(key, self.params[key][i]))

        return '\n'.join(out)

    def fromdir(self, dirfilter='', ffilter=''):
        dir = pyppms.latestdir(dirfilter)
        print('importing data from ' + dir)
        files = fnmatch.filter(os.listdir(dir), '*'+ffilter+'*.dat')
        # find something different about the file names and call that the
        # sample name
        # There's an issue here if you want case insensitivity
        snames = [s.replace(_long_substr(files),'') for s in files]
                #os.mkdir(dirpath)
        self.snames.extend(snames)
        data = [pyppms.ppms(fn) for fn in files]
        self.data.extend(data)
        # todo: load params from file

    def fromfile(self, filter):
        ''' import ppms data from files.  Could make it a classmethod'''
        # could be intelligent about which files are used, so they don't have
        # to be specified one by one
        file = pyppms.latest(filter)
        # name sample relative to others in its directory
        path, fn = os.path.split(file)
        files = fnmatch.filter(os.listdir(path), '*.dat')
        sname = fn.replace(_long_substr(files),'')
        self.snames.append(sname)
        data = pyppms.ppms(file)
        self.data.append(data)


    def write(self, path=None):
        ''' Write data to disk '''
        if path is None:
            path = os.path.join( self.dir, self.name+'.ppms')
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        with open(path, 'w') as f:
            pickle.dump(self, f)
        print('Written to ' + os.path.abspath(path))


    def report(self):
        ''' Generate report on all data contained '''
        pass

    def addsample(self, snames):
        ''' add sample names to container '''
        # TODO: parse this kind of string 's[1-4]'
        try:
            self.snames.extend(sn)
        except:
            self.snames.append(sn)

    #def addparam(self, paramname, values):
        '''
        assign params to snames.
        something like

        self.params += ('paramname', [1,2,3,4])
        self.params[3] = 

        or

        self.params.names = ['param1', 'param2']
        self.params[1,2:6] = [1,2,3,4]
        self.params['param1',:] = [1,2,3,4]

        pandas style.
        maybe you can use pandas.
        '''
        

    def type(self):
        ''' some way to define the type of measurement '''
        pass

    def collect_data(self, filter=None):
        ''' collect data for some or all of the samples based on type '''
        # create directory
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        # create recipe, run it
        # TODO: filename based on type
        filenames = [sn + '_PerpHys.dat' for sn in self.snames]

        for i,fn in enumerate(filenames):
            dpath = os.path.join(self.dir, fn)
            seq.hys(5000, -5000, fn=dpath, rate=100, pts=1000)
            # TODO: run
            # TODO: detect when done, motor friction scan
            c = raw_input('(n)ext / (q)uit?')
            # link data to sname
            try:
                self.data[i] = pyppms.ppms(dpath)
            except:
                # could not find data
                pass

            if c == 'q':
                break

        print('Done')

            # TODO: put parameters into the file

    
    def corr(self, *args, **kwargs):
        self.data = [d.corr(*args, **kwargs) for d in self.data]

    def plot(self, filter=None, label=None, hold=True):
        '''
        calls plot method for each data specified by filter.  Can label by
        parameter
        '''
        if hold:
            for d,sn in zip(self.data, self.snames):
                if not filter or filter in sn:
                    # TODO: should put better filtering
                    d.plot()
                    plt.hold(True)
            plt.title(self.name)
            if label:
                plt.legend(self.params[label], title=label, loc='best')
            else:
                plt.legend(self.snames, loc='best')
        else:
            for d,sn in zip(self.data, self.snames):
                if not filter or filter in sn:
                    # TODO: should put better filtering
                    plt.figure()
                    d.plot()

def _long_substr(data):
    ''' Returns the longest substring contained in data '''
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    return substr
