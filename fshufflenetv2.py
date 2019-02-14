import mxnet as mx
import symbol_utils

def concat_shuffle_split(residual, data, groups=2):
    data = mx.sym.concat(residual, data, dim=1)
    # channel shuffle
    data = channel_shuffle(data, groups)
    # channel split
    data = mx.sym.split(data, axis=1, num_outputs=2)
    return data[0], data[1]

def channel_shuffle(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data

def Activation(data, act_type):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu')
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type)
    return body

def basic_unit(residual, data, out_channels, act_type):
    in_channels = out_channels // 2
    data = mx.sym.Convolution(data=data, num_filter=in_channels, 
                    kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = Activation(data, act_type)

    data = mx.sym.Convolution(data=data, num_filter=in_channels, kernel=(3, 3), 
                    pad=(1, 1), stride=(1, 1), num_group=in_channels)      # depth-wise conv
    data = mx.sym.BatchNorm(data=data)

    data = mx.sym.Convolution(data=data, num_filter=in_channels, 
                    kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)

    return residual, data

def basic_unit_with_downsampling(residual, in_channels, out_channels, act_type):
    data = mx.sym.Convolution(data=residual, num_filter=in_channels, 
                    kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = Activation(data=data, act_type=act_type)

    data = mx.sym.Convolution(data=data, num_filter=in_channels, kernel=(3, 3), 
                    pad=(1, 1), stride=(2, 2), num_group=in_channels)       # depth-wise conv
    data = mx.sym.BatchNorm(data=data)

    data = mx.sym.Convolution(data=data, num_filter=out_channels//2, 
                    kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = Activation(data=data, act_type=act_type)

    residual = mx.sym.Convolution(data=residual, num_filter=in_channels, kernel=(3, 3), 
                    pad=(1, 1), stride=(2, 2), num_group=in_channels)       # depth-wise conv
    residual = mx.sym.BatchNorm(data=residual)
    residual = mx.sym.Convolution(data=residual, num_filter=out_channels//2, 
                    kernel=(1, 1), stride=(1, 1))
    residual = mx.sym.BatchNorm(data=residual)
    residual = Activation(data=residual, act_type=act_type)

    return residual, data

def make_stage(data, stage, depth_multiplier=1, act_type='relu'):
    stage_repeats = [3, 7, 3]

    if depth_multiplier == 0.5:
        out_channels = [-1, 24, 48, 96, 192]
    elif depth_multiplier == 1:
        out_channels = [-1, 24, 116, 232, 464]
    elif depth_multiplier == 1.5:
        out_channels = [-1, 24, 176, 352, 704]
    elif depth_multiplier == 2:
        out_channels = [-1, 24, 244, 488, 976]
       
    residual, data = basic_unit_with_downsampling(data, out_channels[stage-1], 
                                                    out_channels[stage], act_type)

    for i in range(stage_repeats[stage - 2]):
        residual, data = concat_shuffle_split(residual, data, groups=2)
        residual, data = basic_unit(residual, data, out_channels[stage], act_type)

    data = mx.sym.concat(residual, data, dim=1)
    data = channel_shuffle(data, groups=2)

    return data

def get_shufflenet_v2(num_classes=10):  
    depth_multiplier = 1.0      # levels of complexities
    act_type = 'relu'
    fc_type = 'GDC'

    data = mx.symbol.Variable(name="data")
    # data = data-127.5
    # data = data*0.0078125
    data = mx.sym.Convolution(data=data, num_filter=24, 
        	                  kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    # the input size 224x224 --> 112x112, delete this pooling layer
    # data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max', 
    # 	                  stride=(2, 2), pad=(1, 1))
    
    data = make_stage(data, 2, depth_multiplier, act_type)
    data = make_stage(data, 3, depth_multiplier, act_type)
    data = make_stage(data, 4, depth_multiplier, act_type)

    final_channels = 1024 if depth_multiplier != '2.0' else 2048
    data = mx.sym.Convolution(data=data, num_filter=final_channels, 
                              kernel=(1, 1), stride=(1, 1))
    
    # global average pooling
    # data = mx.sym.Pooling(data=data, kernel=(1, 1), global_pool=True, pool_type='avg')
    # data = mx.sym.flatten(data=data)
    # data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)
    # fc1 = mx.sym.SoftmaxOutput(data=data, name='softmax')

    data = symbol_utils.get_fc1(data, num_classes, fc_type)
    fc1 = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return fc1

