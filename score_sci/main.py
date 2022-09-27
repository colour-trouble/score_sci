# @title Autoload all modules

import numpy as np
import scipy.io as sio
import torch
import sampling
import datasets
import numpy as np
import tensorflow as tf
import models
import matplotlib.pyplot as plt
import os
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from models import ncsnpp
from utils import restore_checkpoint
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,)

torch.cuda.empty_cache()

# @title Score model setup (only VESDE for now)

sde = 'vesde'
dataset_id = 'FFHQ'  # Dataset here corresponds to that used for training
# dataset_id = 'CIFAR10'

sequence_len = 8  # Represents the number of frames in the video sequence
iterations = 2000 # Number of sampling iterations == no. of noise scales

if sde.lower() == 'vesde':
    if dataset_id.lower() == 'cifar10':
        # from configs.vp import cifar10_ddpmpp_continuous as configs
        # ckpt_filename = "/home/zhenyuen/Documents/model_checkpoints/vp/ddpm/cifar10_ddpmpp_continuous/checkpoint_26.pth"
        from configs.ve import cifar10_ncsnpp_continuous as configs
        ckpt_filename = "./checkpoints/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    elif dataset_id.lower() == 'ffhq':
        from configs.ve import ffhq_256_ncsnpp_continuous as configs
        ckpt_filename = "./checkpoints/ve/ffhq_256_ncsnpp_continuous/checkpoint_48.pth"

    config = configs.get_config()
    config.training.batch_size = sequence_len  # Required by model
    config.eval.batch_size = sequence_len  # Required by model
    config.sequence_len = sequence_len
    config.model.num_scales = iterations # Overwrite no. iterations here

    sde = VESDE(sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max, N=config.model.num_scales)
    
    sampling_eps = 1e-5

elif sde.lower() == 'vpsde':
    pass

elif sde.lower() == 'subvpsde':
    pass


random_seed = 0  # @param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# @title Visualization


def image_grid(x):
    size = config.data.image_size
    channels = config.data.num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose(
        (0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img


def show_samples(gen, ori, meas, dir_name=None):
    gen = gen.permute(0, 2, 3, 1).detach().cpu().numpy()
    ori = ori.permute(0, 2, 3, 1).detach().cpu().numpy()
    stack = []
    # Number of images per row is hard coded to 4, change if necessary
    stack.append(ori[0:4])
    # This is done as all testing samples thus far are 8-frame video sequences.
    stack.append(gen[0:4])
    stack.append(ori[4:8])
    stack.append(gen[4:8])
    x = np.concatenate(stack, axis=0)
    img = image_grid(x)

    if dir_name:
        # Save measured sample
        if not os.path.exists(f"assets/"):
            os.mkdir("assets/")
            
        if not os.path.exists(f"assets/{dir_name}"):
            os.mkdir(f"assets/{dir_name}")

        meas = meas.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(meas / 8)
        plt.savefig(f"assets/{dir_name}/measured.png")

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(img)

    if dir_name:
        plt.savefig(f"assets/{dir_name}/generated.png")

    plt.show()

# @title Load original and measured video sequence from dataset


def load_dataset(scene, config):
    img_size = config.data.image_size
    channels = config.data.num_channels
    sequence_len = config.sequence_len
    shape = (sequence_len, channels, img_size, img_size)

    def preprocess_fn(f):
        def process(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(
                img, (config.data.image_size, config.data.image_size), antialias=True)
            img = tf.transpose(img, (2, 0, 1))
            return img

        def parse(f):
            img = tf.io.read_file(f)  # parse
            img = tf.image.decode_png(img, channels=3)  # 3 For colour
            img = process(img)
            return img

        return parse(f)

    def get_images():
        ds = tf.data.Dataset.list_files(
            f'./datasets/davis/{scene}/*', shuffle=False)
        ds = ds.map(preprocess_fn)
        ori = [img for img in ds.take(8)]
        ori = tf.stack(ori, axis=0)
        assert ori.shape == (8, 3, config.data.image_size,
                             config.data.image_size)
        return ori.numpy()

    def get_rgb_mask():
        # path of the .mat data filefile = sio.loadmat(matfile, appendmat=True) # for '-v7.2' and lower version of .mat file (MATLAB)
        matfile = './datasets/cacti/crash32_cacti.mat'
        # for '-v7.2' and lower version of .mat file (MATLAB)
        file = sio.loadmat(matfile, appendmat=True)
        mask = np.float32(file['mask'])
        mask = mask[None, ...]
        mask = np.repeat(mask, 3, 0)
        mask = np.transpose(mask, [3, 0, 1, 2])
        mask = mask[:, :, 0:config.data.image_size, 0:config.data.image_size]
        assert mask.shape == (8, 3, config.data.image_size,
                              config.data.image_size)
        return mask

    def get_meas(ori, mask):
        meas = np.zeros(
            shape=(3, config.data.image_size, config.data.image_size))
        for i in range(ori.shape[0]):
            meas += mask[i, :, :, :] * ori[i, :, :, :]
        return meas

    ori = get_images()
    mask = get_rgb_mask()
    meas = get_meas(ori, mask)

    meas = torch.from_numpy(meas).to(device=config.device)
    mask = torch.from_numpy(mask).to(device=config.device)
    ori = torch.from_numpy(ori).to(device=config.device)

    return (meas, mask, ori)

# @title Setup sampling function


def get_sampler(sampler, config):
    # Only this predictor/corrector pair tested thus far
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector

    snr = 0.16  # @param {"type": "number"}
    n_steps = 1  # @param {"type": "integer"}
    probability_flow = False  # @param {"type": "boolean"}

    img_size = config.data.image_size
    channels = config.data.num_channels
    sequence_len = config.sequence_len
    shape = (sequence_len, channels, img_size, img_size)

    if sampler.lower() == 'cond_rev_sde':
        sampling_fn = sampling.get_cond_rev_sde(sde, shape, predictor, corrector,
                                                inverse_scaler, snr, n_steps=n_steps,
                                                probability_flow=probability_flow,
                                                continuous=config.training.continuous,
                                                eps=sampling_eps, device=config.device)

    elif sampler.lower() == 'prox_op':
        coeff = 0.9 # Balancing hyper-parameter
        sampling_fn = sampling.get_prox_op_sampler(sde, shape, predictor, corrector,
                                                inverse_scaler, snr, n_steps=n_steps,
                                                probability_flow=probability_flow,
                                                continuous=config.training.continuous,
                                                eps=sampling_eps, device=config.device, coeff=coeff)

    elif sampler.lower() == 'proj_exp':
        # Projecting expectation method -- under Section 5.4 "Other variations"
        coeff = 0.9 # Balancing hyper-parameter
        sampling_fn = sampling.get_proj_exp_sampler(sde, shape, predictor, corrector,
                                                inverse_scaler, snr, n_steps=n_steps,
                                                probability_flow=probability_flow,
                                                continuous=config.training.continuous,
                                                eps=sampling_eps, device=config.device, coeff=coeff)        

    elif sampler.lower() == 'vid_frame_approx':
        coeff = 0.9 # Balancing hyper-parameter
        sampling_fn = sampling.get_vid_frame_approx_sampler(sde, shape, predictor, corrector,
                                                inverse_scaler, snr, n_steps=n_steps,
                                                probability_flow=probability_flow,
                                                continuous=config.training.continuous,
                                                eps=sampling_eps, device=config.device, coeff=coeff)

    return sampling_fn

# Select scene to reconstruct under /datasets/davis/
# scene = 'walking'
scene = 'bear'
# scene = 'boat'
# scene = 'dog'
# scene = 'train'

# Select sampler
sampler = 'cond_rev_sde'  # Conditional reverse-time SDE method
# sampler = 'prox_op' # Proximal optimization method
# sampler = 'proj_exp' # Projecting expectation method
# sampler = 'vid_frame_approx' # Video frame approximation method


sampling_fn = get_sampler(sampler, config)
meas, mask, ori = load_dataset(scene, config)

sample = sampling_fn(score_model, x=ori, y=meas, mask=mask)

# Pass dir_name argument to save sample under assets/dir_name
dir_name = f"{scene}_{sampler}"
show_samples(sample, ori, meas, dir_name=dir_name)
