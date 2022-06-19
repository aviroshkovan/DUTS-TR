import os
import numpy as np
import cv2
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
import wandb
from wandb.keras import WandbCallback

wandb.init(project="ModelSeg", entity="aviroshkovan")

np.random.seed(42)
tf.random.set_seed(42)

#Initializing parameters
IMAGE_SIZE = 256
EPOCHS = 30
BATCH = 16
LR = 1e-4

BASE_OUTPUT = '/mnt/md0/mory/DUTS-TR/'
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,
	"loss1_plot.png"])

images_path ='/mnt/md0/mory/DUTS-TR/DUTS-TR-Image'
mask_path = '/mnt/md0/mory/DUTS-TR/DUTS-TR-Mask'

#Sorting the names of the files
images_names = sorted(
    [
        os.path.join(images_path, fname)
        for fname in os.listdir(images_path)
        if fname.endswith(".jpg")
    ]
)
mask_names = sorted(
    [
        os.path.join(mask_path, fname)
        for fname in os.listdir(mask_path)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print(len(images_names))
print(len(mask_names))

np.random.seed(42)
tf.random.set_seed(42)

#Load data
print("Loading Images...")

#Capture training image info as a list
train_images = []

for directory_path in glob.glob(images_path):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img/255.0   
        train_images.append(img)
        
train_images = np.array(train_images)        
print(train_images.shape)

print("Loading Masks...")

np.random.seed(42)
tf.random.set_seed(42)

#Capture mask/label info as a list
train_masks = [] 

for directory_path in glob.glob(mask_path):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.expand_dims(mask, axis = -1)
        mask = mask/255.0
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)
print(train_masks.shape)

np.random.seed(42)
tf.random.set_seed(42)

#Picking 20% for testing and remaining for training
train_x, test_x, train_y, test_y = train_test_split(train_images, train_masks, test_size = 0.20, random_state = 42)

#Checking the shapes after the split
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

np.random.seed(42)
tf.random.set_seed(42)

#Defining model architecture 
def model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

#Defining loss function 
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

model = model()
model.summary()   

np.random.seed(42)
tf.random.set_seed(42)

#Compiling the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(LR)
metrics = [dice_coef, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

#Defining callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    WandbCallback()
]

np.random.seed(42)
tf.random.set_seed(42)

print("[INFO] training model...")
history=model.fit(
    train_x, 
    train_y,
    batch_size=BATCH, 
    epochs=EPOCHS,
    verbose=1,
    validation_data=(test_x, test_y),
    callbacks=callbacks)

print("[INFO] saving model...")
model.save("model_seg1.h5")

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["dice_coef"], label="train_acc")
    plt.plot(H.history["val_dice_coef"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

plot_training(history, PLOT_PATH)    
