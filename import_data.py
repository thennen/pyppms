# I don't remember what the heck I wrote here years ago
# I just want a function that loads any type of ppms file, so here it is.

import re

def importdata(filepath):
    # Read header and locate start of data
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
        colnames = next(fenum)[1].split(',')

    # This is dumb but whatever
    cols = []
    units = {}
    for c in colnames:
        clean = c.strip().replace(' ', '').replace('\'', '').replace('.','')
        rs = re.split('\((.*)\)', clean)
        cols.append(rs[0])
        if len(rs) > 1:
            units[rs[0]] = rs[1]

    df = pd.read_csv(filepath, skiprows=skip-1)
    df.columns = cols
    df = df.dropna(1, how='all')
    # Could return it like this

    data = dict(header=header, units=units)
    for k,v in df.items():
        data[k] = v.values

    return pd.Series(data)

    #np.genfromtxt(filepath, skip_header=skip, delimiter=',', unpack=True, usecols=(2,3,4))
