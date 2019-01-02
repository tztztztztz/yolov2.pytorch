debug = False

batch_size = 32

lr = 0.001

decay_lrs = {
    60: 0.0001,
    90: 0.00001
}

momentum = 0.9
weight_decay = 0.0005


input_size = (416, 416)
output_size = (13, 13)

strides = 32

anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

thresh = .6

