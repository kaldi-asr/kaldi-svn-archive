import sys

DIROUT = sys.argv[1]

lines = sys.stdin.readlines()

def gettimes(frames, framesz = 0.01):
    times = []
    t = 0.0
    frames = frames.replace(' ]', '')
    frames = frames.split(' [')
    for p in frames[1:]:
        pp = p.strip().split()
        dur = len(pp) * framesz
        times.append("%.3f %.3f" % (t, t + dur))
        t  += dur
    return times

for lno in range(0, len(lines), 3):
    times = gettimes(lines[lno])
    labs = lines[lno + 1].split()
    uttid = labs[0]
    labs = labs[1:]

    fp = open(DIROUT + '/' + uttid + '.lab', 'w')
    for i in range(len(labs)):
        fp.write("%s %s\n" % (times[i], labs[i].split('_')[0]))
    fp.close()
    
