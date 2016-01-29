def beep():
    cmd = 'BEP BEEP 0.10 500\n'
    return cmd

def callseq(path):
    cmd = 'CALL {}\n'.format(path)
    return cmd

def purgeseal():
    cmd = 'CMB CHAMBER 1\n'
    return cmd

def ventseal():
    cmd = 'CMB CHAMBER 2\n'
    return cmd

def remark(msg):
    cmd = 'REM {}\n'.format(msg)
    return cmd

def scanfield():
    pass

def scantemp():
    pass

def scantime():
    pass

def sendmsg():
    ''' Sends you an email '''
    cmd = 'MES 1.000000 0 \"message\", \"tyler@hennen.us\", \"subject\", \"\", \"message\"\n'
    return cmd
    
def setfield(Oe, rate):
    cmd = 'FLD FIELD {:.1f} {:.1f} 0 0\n'.format(Oe,rate)
    return cmd

def settemp(T, kpermin):
    cmd = 'TMP TEMP {:.6f} {:.6f} 0\n'.format(T, kpermin)
    return cmd

def logdata(path,title='',comment=''):
    cmd = 'LOG 0 1 1.00 254 0 0 \"{:s}\" \"{:s}\" \"{:s}\" \"\"\n'.format(path, title, comment)
    return cmd

def wait(sec=0):
    ''' waits for system stability + time '''
    cmd = 'WAI WAITFOR {:d} 0 1 0 1 0\n'.format(int(sec))
    return cmd

def meas(sec):
    ''' Measure for some time '''
    #TODO: make this work
    cmd = 'VSMCO 40 0 0 0 0 2 40 1 0 2 0 \"A/C,1,10,10,0\"\n'
    return cmd

def touchdown():
    cmd = 'VSMLS 1 0 0 0 0 0\n'
    return cmd

def comment():
    cmd = 'VSMCM \"datafile comment\"\n'
    return cmd

def MH(posfield, negfield, rate=200., pts=500.):
    ''' M H loop '''
    #TODO: add options, limit rate
    avgtime = 2*(posfield-negfield)/float(rate)/float(pts)
    # no autocenter
    cmd = 'VSMMH 1 0 0 0 0 2 40 {:.2f} 0 2 0 2 6 {:d} 0 {:d} {:d} 1 25 50 0 2 1 0 1 0 \"A/C,0,10,10,0\"\n'.format(avgtime, int(negfield), int(posfield), int(rate))
    return cmd

def MT(temp1, temp2, rate):
    ''' M T loop '''
    #TODO make sure this works
    cmd = 'VSMMT 1 0 0 0 0 2 40 1 0 2 0 {:d} {:d} 1 1 25 50 0 2 0 1 0 \"A/C,1,10,10,0\"\n'.format(int(postemp), int(negtemp))
    return cmd

def datafile(filepath,append=False):
    ''' Creates new/ appends to data file  '''
    #TODO: check if the data file is already there, decide what to do
    cmd = 'VSMDF \"{:s}\" 0 {:b} \"\"\n'.format(filepath,append)
    return cmd


