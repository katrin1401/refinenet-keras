import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

from scripts.customGenerator import customGenerator
from scripts.helpers import get_label_info, gen_dirs, save_settings
from scripts.training import step_decay_schedule, ignore_unknown_xentropy, TensorBoardWrapper
from scripts.customCallbacks import LossHistory, OutputObserver

from model.refinenet import build_refinenet

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import extra_keras_metrics
from extra_keras_metrics import average_precision_at_k, mean_iou

#### Define parameters
# Dataset
dataset_basepath = r'C:\Users\KatrinKostova\OneDrive - Brightcape BV\Documents\Scriptie docus\imaterialist-fashion-2019-FGVC6\RefineNet\v11'
train_images = os.path.join(dataset_basepath,'training/images')
train_masks = os.path.join(dataset_basepath,'training/labels')
val_images = os.path.join(dataset_basepath,'validation/images')
val_masks = os.path.join(dataset_basepath,'validation/labels')
# class_dict = r'C:\Users\KatrinKostova\OneDrive - Brightcape BV\Documents\Scriptie docus\imaterialist-fashion-2019-FGVC6\RefineNet\v8\class_dict.csv'
class_dict = 'class_dict.csv'

# ResNet
frontend_weights = r'C:\Users\KatrinKostova\OneDrive - Brightcape BV\Documents\Scriptie docus\imaterialist-fashion-2019-FGVC6\RefineNet\resnet101_weights_tf.h5'
frontend_trainable = True

# Input dimensions (height, width, channels)
input_shape = (384,384,3)
random_crop = None # or None if no random cropping required

# Train settings
# pre_trained_weights = None
pre_trained_weights = r'C:\Users\KatrinKostova\OneDrive - Brightcape BV\Documents\Github\refinenet-keras\runs\20200424-150017\weights/weights.02-0.20.hdf5'
batch_size = 4
epochs = 25
lr_init = 0.001 # 1e-4
validation_images = 4
aug_dict = {'rotation_range': 10, 
            'height_shift_range': 0.1,
            'width_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': (0.7, 1.3)}

# aug_dict1 = {'rotation_range': 0,
#             'height_shift_range': 0,
#             'width_shift_range': 0,
#             'shear_range': 0,
#             'zoom_range': 0,
#             'horizontal_flip': False,
#             'vertical_flip': False,
#             'brightness_range': (0, 0)}
steps_per_epoch = 10
lrate = step_decay_schedule(initial_lr = lr_init, decay_factor = 0.1, step_size = 25)

# Import classes from csv file
mask_colors, num_class = get_label_info(os.path.join(dataset_basepath,class_dict))
   
dirs = gen_dirs()

# Data generators for training
myTrainGen = customGenerator(batch_size, train_images, train_masks, num_class, input_shape, aug_dict, mask_colors, random_crop = random_crop)
myValGen = customGenerator(batch_size, val_images, val_masks, num_class, input_shape, dict(), mask_colors, random_crop = random_crop)

# Define callbacks
model_checkpoint = ModelCheckpoint(
        dirs['weight_dir']+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor = 'val_loss', verbose = 1, save_best_only = True)
tbCallBack = TensorBoardWrapper(myValGen.generator(), validation_images // batch_size,
                                batch_size, log_dir=dirs['run_dir'], histogram_freq=0,
                                write_graph=True, write_grads=True,
                                batch_size=batch_size, write_images=False)

history = LossHistory(dirs['batch_log_path'], dirs['epoch_log_path'])

tmp_data = next(myValGen.generator())[0]
save_imgs = OutputObserver(tmp_data, dirs['img_dir'], mask_colors)

# Build and compile RefineNet
input_shape = random_crop if random_crop else input_shape # adjust network input for random cropping
model = build_refinenet(input_shape, num_class, resnet_weights = frontend_weights, frontend_trainable = frontend_trainable)
# meanIoU = tf.keras.metrics.MeanIoU(num_classes=num_class)
# model.compile(optimizer = Adam(lr=lr_init), loss = ignore_unknown_xentropy, metrics = ["true_positives", "true_negatives", "false_positives", "false_negatives", "accuracy"])
model.compile(optimizer = Adam(lr=lr_init), loss = ignore_unknown_xentropy, metrics = ["accuracy"])

if pre_trained_weights:
    model.load_weights(pre_trained_weights)

save_settings(dirs['settings_path'],
                dirs['summary_path'],
                frontend_trainable = frontend_trainable,
                datagen_args = aug_dict,
                batch_size = batch_size,
                input_shape = input_shape,
                dataset_basepath = dataset_basepath,
                frontend_weights = frontend_weights,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                pre_trained_weights = pre_trained_weights,
                model = model,
                lr_init = lr_init)

# Start training
# %load_ext tensorboard
from tensorflow import keras
from time import localtime, strftime
date_time = strftime("%Y%m%d-%H%M%S", localtime())
runs = r"C:\Users\KatrinKostova\OneDrive - Brightcape BV\Documents\Github\refinenet-keras\runs"
logdir = os.path.join(runs, date_time)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.fit_generator(myTrainGen.generator(),
                    steps_per_epoch = steps_per_epoch,
                    validation_data = myValGen.generator(),
                    validation_steps = validation_images // batch_size,
                    epochs = epochs,
                    callbacks = [model_checkpoint, tbCallBack, lrate, history, save_imgs, tensorboard_callback])

