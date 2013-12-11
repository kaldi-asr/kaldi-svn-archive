# Uses python wrappers to print out a context dependency tree


import os, sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# load idlakapi wrapper
sys.path += [os.path.join(SCRIPT_DIR, 'pythonlib')]
import idlakapi


class KaldiTree:
    def __init__(self, treedata):
        pass


# only works for binary trees
def printevent(event, eventvector, buf):
    # get children
    idlakapi.IDLAK_eventmap_getchildren(event, eventvector)
    # terminal node
    if not idlakapi.IDLAK_eventmapvector_size(eventvector):
        print 'CE', idlakapi.IDLAK_eventmap_answer(event),
    else:
        yes = idlakapi.IDLAK_eventmapvector_at(eventvector, 0)
        no = idlakapi.IDLAK_eventmapvector_at(eventvector, 1)
        idlakapi.IDLAK_eventmap_yesset(event, buf)
        yesset = '[ ' + idlakapi.IDLAK_string_val(buf) + ']'
        print 'SE', idlakapi.IDLAK_eventmap_key(event), yesset
        print '{',
        printevent(yes, eventvector, buf)
        printevent(no, eventvector, buf)
        print '} '
        

def main():
    from optparse import OptionParser

    usage="Usage: %prog [-h] -t kalditree\n\nPrint kaldi decision tree in ascii"
    parser = OptionParser(usage=usage)
    # Options
    parser.add_option("-t", "--kalditree", default=None,
                      help="Kaldi tree")
    opts, args = parser.parse_args()

    if not opts.kalditree:
        parser.error("Require input kaldi tree")
    # convert to ascii and load as a string
    context_tree = idlakapi.IDLAK_read_contextdependency_tree(opts.kalditree)
    print "ContextDependency",
    print idlakapi.IDLAK_contextdependency_tree_contextwidth(context_tree),
    print idlakapi.IDLAK_contextdependency_tree_centralposition(context_tree),
    print "ToPdf",
    root = idlakapi.IDLAK_contextdependency_tree_root(context_tree)
    eventvector = idlakapi.IDLAK_eventmapvector_new()
    buf = idlakapi.IDLAK_string_new()
    printevent(root, eventvector, buf)
    idlakapi.IDLAK_eventmapvector_delete(eventvector)
    idlakapi.IDLAK_string_delete(buf)
    print "EndContextDependency ",
    
if __name__ == '__main__':
    main()
