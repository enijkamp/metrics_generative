import numpy as np
from fid_v2_tf import fid_score
from cifar10_tf import input_fn

#############################################################################################################
# fid score with tensorflow data loader
#############################################################################################################

def create_session():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.visible_device_list = str(0)
    return tf.Session(config=config)


with create_session() as sess:
    dataset = input_fn(True, '/tmp/cifar10_data', 50000)
    one_iter = dataset.make_one_shot_iterator()
    data = list(sess.run(one_iter.get_next()[0]))
    dataset_full = np.array([x for x in data])

    dataset = input_fn(True, '/tmp/cifar10_data', 1000)
    one_iter = dataset.make_one_shot_iterator()
    samples = list(sess.run(one_iter.get_next()[0]))
    gen_samples_list = np.array([x for x in samples])


print('fid with tensorflow dataset')
fid1 = fid_score(create_session, dataset_full, gen_samples_list)
print(fid1)
# 29.852803521371698
assert fid1 < 32

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

print('fid with pytorch dataset')
fid2 = fid_score(create_session, dataset_nhwc, gen_samples_list)
print(fid2)
# 29.85279935346972
assert fid2 < 32
