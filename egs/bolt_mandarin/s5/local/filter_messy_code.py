#remove abnormal characters in lexcion
import re,sys

p = re.compile("[^A-Za-z0-9]+ [A-Za-z0-9: ]+$")
for l in sys.stdin:
    m = p.match(l)
    if m:
        print l,

