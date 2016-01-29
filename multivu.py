import os

class seq():
    ''' Wrapper for retarded PPMS sequence language '''
    def __init__(self):
        self.lines = []

    def write(self, path):
        ''' Write the sequence to disk '''
        self.path = path
        with open(self.path, 'w') as f:
            f.writelines(self.lines)

    def open(self):
        ''' Open the written seq file'''
        os.startfile(self.path)

    # These are PPMS commands

    def beep(self):
        cmd = 'BEP BEEP 0.10 500\n'
        self.lines.append(cmd)

    def callseq(self, path):
        cmd = 'CALL {}\n'.format(path)
        self.lines.append(cmd)

    def purgeseal(self):
        cmd = 'CMB CHAMBER 1\n'
        self.lines.append(cmd)

    def ventseal(self):
        cmd = 'CMB CHAMBER 2\n'
        self.lines.append(cmd)

    def remark(self, msg):
        cmd = 'REM {}\n'.format(msg)
        self.lines.append(cmd)

    def scanfield(self):
        pass

    def scantemp(self):
        pass

    def scantime(self):
        pass

    def sendmsg(self):
        ''' Sends you an email '''
        cmd = 'MES 1.000000 0 \"message\", \"tyler@hennen.us\", \"subject\", \"\", \"message\"\n'
        self.lines.append(cmd)

    def setfield(self, Oe, rate):
        cmd = 'FLD FIELD {:.1f} {:.1f} 0 0\n'.format(Oe,rate)
        self.lines.append(cmd)

    def settemp(self, T, kpermin):
        cmd = 'TMP TEMP {:.6f} {:.6f} 0\n'.format(T, kpermin)
        self.lines.append(cmd)

    def logdata(self, path, title='',comment=''):
        cmd = 'LOG 0 1 1.00 254 0 0 \"{:s}\" \"{:s}\" \"{:s}\" \"\"\n'.format(path, title, comment)
        self.lines.append(cmd)

    def wait(self, sec=0):
        ''' waits for system stability + time '''
        cmd = 'WAI WAITFOR {:d} 0 1 0 1 0\n'.format(int(sec))
        self.lines.append(cmd)

    def meas(self, sec):
        ''' Measure for some time '''
        #TODO: make this work
        cmd = 'VSMCO 40 0 0 0 0 2 40 1 0 2 0 \"A/C,1,10,10,0\"\n'
        self.lines.append(cmd)

    def touchdown(self):
        cmd = 'VSMLS 1 0 0 0 0 0\n'
        self.lines.append(cmd)

    def comment(self):
        cmd = 'VSMCM \"datafile comment\"\n'
        self.lines.append(cmd)

    def MH_loop(self, posfield, negfield, rate=200., pts=500.):
        ''' M H loop '''
        avgtime = 2*(posfield-negfield)/float(rate)/float(pts)
        # no autocenter
        cmd = 'VSMMH 1 0 0 0 0 2 40 {:.2f} 0 2 0 2 6 {:d} 0 {:d} {:d} 1 25 50 0 2 1 0 1 0 \"A/C,0,10,10,0\"\n'.format(avgtime, int(negfield), int(posfield), int(rate))
        self.lines.append(cmd)

    def MH(self, startfield, endfield, rate=40., pts=500.):
        avgtime = abs(2*(endfield-startfield)/float(rate)/float(pts))
        midfield = (startfield + endfield) / 2
        if startfield < endfield:
            start = 4
            end = 6
        else:
            start = 2
            end = 4
        cmd = 'VSMMH 1 0 0 0 0 2 40 {:.2f} 0 2 0 {:d} {:d} {:d} {:d} {:d} {:d} 1 25 50 0 2 1 0 1 0 \"A/C,0,10,10,0\"\n'.format(avgtime, start, end, int(startfield), int(midfield), int(endfield), int(rate))
        self.lines.append(cmd)
        # 1 0 0 0 0 2 40 avgtime 0 2 0 startquad endquad startfield 0 endfield
        # rate 1 25 50 0 2 1 0 1 0 blahblah


    def MT(self, temp1, temp2, rate, ):
        ''' M T loop '''
        #TODO make sure this works
        cmd = 'VSMMT 1 0 0 0 0 2 40 1 0 2 0 {:d} {:d} 1 1 25 50 0 2 0 1 0 \"A/C,1,10,10,0\"\n'.format(int(postemp), int(negtemp))
        self.lines.append(cmd)

    def datafile(self, filepath,append=False):
        ''' Creates new/ appends to data file  '''
        #TODO: check if the data file is already there, decide what to do
        cmd = 'VSMDF \"{:s}\" 0 {:b} \"\"\n'.format(filepath,append)
        self.lines.append(cmd)
