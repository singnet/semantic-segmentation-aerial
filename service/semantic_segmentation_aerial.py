import numpy as np
from skimage import io
from tqdm import tqdm_notebook as tqdm
import itertools
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import logging

logging.basicConfig(
    level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s"
)
log = logging.getLogger("semantic_segmentation_aerial")

WINDOW_SIZE = (256, 256)  # Patch size
STRIDE = 32  # Stride for testing
LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        return x


class SemanticSegmentationAerial:

    def __init__(self):
        self.MODEL_PATH = '/opt/singnet/semantic-segmentation-aerial/service/segnet_final_reference.pth'
        self.IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
        self.BATCH_SIZE = 10  # Number of samples in a mini-batch
        self.WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
        self.CACHE = True  # Store the dataset in-memory
        self.palette = {0: (255, 255, 255),  # Impervious surfaces (white)
                        1: (0, 0, 255),  # Buildings (blue)
                        2: (0, 255, 255),  # Low vegetation (cyan)
                        3: (0, 255, 0),  # Trees (green)
                        4: (255, 255, 0),  # Cars (yellow)
                        5: (255, 0, 0),  # Clutter (red)
                        6: (0, 0, 0)}  # Undefined (black)
        self.invert_palette = {v: k for k, v in self.palette.items()}

        self.net = SegNet()

    def convert_to_color(self, arr_2d):
        """ Numeric labels to RGB-color encoding """
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in self.palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    def convert_from_color(self, arr_3d):
        """ RGB-color encoding to grayscale labels """
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in self.palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    # Utils
    @staticmethod
    def sliding_window(top, step, sliding_window_size):
        """ Slide a window_shape window across the image with a stride of step """
        for x in range(0, top.shape[0], step):
            if x + sliding_window_size[0] > top.shape[0]:
                x = top.shape[0] - sliding_window_size[0]
            for y in range(0, top.shape[1], step):
                if y + sliding_window_size[1] > top.shape[1]:
                    y = top.shape[1] - sliding_window_size[1]
                yield x, y, sliding_window_size[0], sliding_window_size[1]

    @staticmethod
    def count_sliding_window(top, step):
        """ Count the number of windows in an image """
        c = 0
        for x in range(0, top.shape[0], step):
            # if x + sliding_window_size[0] > top.shape[0]:
            #     x = top.shape[0] - sliding_window_size[0]
            for y in range(0, top.shape[1], step):
                # if y + sliding_window_size[1] > top.shape[1]:
                #     y = top.shape[1] - sliding_window_size[1]
                c += 1
        return c

    @staticmethod
    def grouper(n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def load_weights(self):
        try:
            self.net.load_state_dict(torch.load(self.MODEL_PATH))
            print("Loaded weights in SegNet !")
        except Exception as e:
            log.error(e)
            raise e
        self.net.cuda()

    def segment(self, input_image, window_size, stride):
        batch_size = self.BATCH_SIZE
        window_size = (window_size, window_size)
        img = (1 / 255 * np.asarray(io.imread(input_image), dtype='float32'))

        # Switch the network to inference mode
        self.net.eval()

        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = self.count_sliding_window(img, stride) // batch_size
        for i, coords in enumerate(tqdm(self.grouper(batch_size,
                                        self.sliding_window(img,
                                                            stride,
                                                            window_size)),
                                        total=total,
                                        leave=False)):
            # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #     Keep progress code ?
            # _pred = np.argmax(pred, axis=-1)

            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            outs = self.net(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x+w, y:y+h] += out
            del outs

        pred = np.argmax(pred, axis=-1)

        return pred

# Parameters

# MODEL_PATH = '/opt/singnet/semantic-segmentation-aerial/service/segnet_final_reference.pth'
# IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
# BATCH_SIZE = 10  # Number of samples in a mini-batch
# LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names

# N_CLASSES = len(LABELS)  # Number of classes
# WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
# CACHE = True  # Store the dataset in-memory

# ISPRS color palette

# Let's define the standard ISPRS color palette
# palette = {0: (255, 255, 255),  # Impervious surfaces (white)
#            1: (0, 0, 255),  # Buildings (blue)
#            2: (0, 255, 255),  # Low vegetation (cyan)
#            3: (0, 255, 0),  # Trees (green)
#            4: (255, 255, 0),  # Cars (yellow)
#            5: (255, 0, 0),  # Clutter (red)
#            6: (0, 0, 0)}  # Undefined (black)
# invert_palette = {v: k for k, v in palette.items()}



# instantiate the network
# try:
#     from urllib.request import URLopener
# except ImportError:
#     from urllib import URLopener
#
# # Download VGG-16 weights from PyTorch
# vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
# if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
#     weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')
#
# vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
# mapped_weights = {}
# for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
#     if "features" in k_vgg:
#         mapped_weights[k_segnet] = vgg16_weights[k_vgg]
#         print("Mapping {} to {}".format(k_vgg, k_segnet))
#
# try:
#     net.load_state_dict(mapped_weights)
#     print("Loaded VGG-16 weights in SegNet !")
# except:
#     # Ignore missing keys
#     pass

# net.cuda()

# TODO: ACTUAL PREDICTION CODE

# input_image = "/root/Shared/aerial.png"
# input_image = "/root/Shared/vie1.jpg"
# STRIDE = 16
# stride = STRIDE
# batch_size = BATCH_SIZE
# WINDOW_SIZE = (256, 256) # Patch size
# window_size = (256, 256)

# def test(net, input_image, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):

# Use the network on the test set
# img = (1 / 255 * np.asarray(io.imread(input_image), dtype='float32'))
# all_preds = []

# Switch the network to inference mode
# net.eval()

# pred = np.zeros(img.shape[:2] + (N_CLASSES,))

# total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
# for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
#     # Display in progress results
#     # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
#     #     # TODO: Remove plotting / Keep progress code ?
#     # _pred = np.argmax(pred, axis=-1)
#
#     # Build the tensor
#     image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in coords]
#     image_patches = np.asarray(image_patches)
#     image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
#
#     # Do the inference
#     outs = net(image_patches)
#     outs = outs.data.cpu().numpy()
#
#     # Fill in the results array
#     for out, (x, y, w, h) in zip(outs, coords):
#         out = out.transpose((1, 2, 0))
#         pred[x:x+w, y:y+h] += out
#     del outs
#
# pred = np.argmax(pred, axis=-1)
