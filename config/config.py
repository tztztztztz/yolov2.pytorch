debug = False

batch_size = 32

lr = 0.001

decay_lrs = {
    1: 0.0001,
    10: 0.001,
    60: 0.0001,
    90: 0.00001
}

momentum = 0.9
weight_decay = 0.0005


input_size = (416, 416)
output_size = (13, 13)

strides = 32

anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

saturation = 1.5
exposure = 1.5
hue=.1

thresh = .6

