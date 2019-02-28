import logging
import mxnet as mx
import numpy as np
from symbol import get_shufflenet
from fshufflenetv2 import get_shufflenet_v2
from fmobilefacenet import get_mobilefacenet

# shufflenet = get_shufflenet()
shufflenet = get_shufflenet_v2()
# shufflenet = get_mobilefacenet()

# save as symbol
sym = shufflenet

## plot network graph
mx.viz.print_summary(sym, shape={'data':(8,3,28,28)})
mx.viz.plot_network(sym, shape={'data':(8,3,28,28)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()