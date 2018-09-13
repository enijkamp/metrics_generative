import numpy as np
from inception_score_v2_tf import get_inception_score
from cifar10_tf import input_fn

#############################################################################################################
# inception score with tensorflow data loader
#############################################################################################################

def create_session():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = str(0)
    return tf.Session(config=config)

print('inception v2 with tensorflow dataset')

with create_session() as sess:
    dataset = input_fn(True, '/tmp/cifar10_data', 50000)
    iter = dataset.make_one_shot_iterator()
    data = list(sess.run(iter.get_next()[0]))

    incept1 = get_inception_score(data, bs=1)
    print(incept1)
    assert incept1[0] < 11.5

    iter = dataset.make_one_shot_iterator()
    incept2 = get_inception_score(data, bs=100)
    print(incept2)
    assert incept2[0] < 11.5

#############################################################################################################
# fid score with pytorch data loader
#############################################################################################################

from torchvision import datasets, transforms
unnormalize = lambda x: x / 2.0 + 0.5
to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
dataset = datasets.CIFAR10(root='./training_images', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
dataset_nhwc = np.clip(255 * to_nhwc(unnormalize(dataset_full)), 0.0, 255)

print('inception v2 with pytorch dataset')
iter = dataset.make_one_shot_iterator()
incept3 = get_inception_score(dataset_nhwc, bs=100)
print(incept3)
assert incept3[0] < 11.5
