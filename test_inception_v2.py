import numpy as np
from inception_score_v2_tf import get_inception_score
from cifar10_tf import input_fn

#############################################################################################################
# inception score with tensorflow data loader
#############################################################################################################

def create_session():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.visible_device_list = str(0)
    return tf.Session(config=config)

print('inception v2 with tensorflow dataset')

with create_session() as sess:
    dataset = input_fn(True, '/tmp/cifar10_data', 50000)
    iter_one = dataset.make_one_shot_iterator()
    data = list(sess.run(iter_one.get_next()[0]))

incept1 = get_inception_score(create_session, data, bs=1)
print(incept1)
# (11.2402725, 0.20432183)
assert incept1[0] > 11

incept2 = get_inception_score(create_session, data, bs=100)
print(incept2)
# (11.24027, 0.20432393)
assert incept2[0] > 11

#############################################################################################################
# inception score with pytorch data loader
#############################################################################################################

from torchvision import datasets, transforms
unnormalize = lambda x: x / 2.0 + 0.5
to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

dataset = datasets.CIFAR10(root='data/', download=True,
                     transform=transforms.Compose([
                         transforms.Scale(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]))

dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
dataset_nhwc = np.clip(255 * to_nhwc(unnormalize(dataset_full)), 0.0, 255.0)
dataset_nhwc_list = [dataset_nhwc[i, :, :, :] for i in range(len(dataset))]

print('inception v2 with pytorch dataset')
incept3 = get_inception_score(create_session, dataset_nhwc_list, bs=100)
print(incept3)
# (11.237376, 0.11622817)
assert incept3[0] > 11

#############################################################################################################

dataset = datasets.CIFAR10(root='data/', download=True,
                     transform=transforms.Compose([
                         transforms.Scale(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                     ]))

dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
dataset_nhwc = np.clip(255 * to_nhwc(unnormalize(dataset_full)), 0.0, 255.0)
dataset_nhwc_list = [dataset_nhwc[i, :, :, :] for i in range(len(dataset))]

print('inception v2 with pytorch dataset')
incept3 = get_inception_score(create_session, dataset_nhwc_list, bs=100)
print(incept3)
(9.877339, 0.13163306)
assert incept3[0] > 9.8
