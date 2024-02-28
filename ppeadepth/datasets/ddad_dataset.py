import dgp
from dgp.datasets import SynchronizedSceneDataset

import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image  # using pillow-simd for increased speed

DDAD_TRAIN_VAL_JSON_PATH = '/mnt/disk-2/alice/ddad/ddad_train_val/ddad.json'
DATUMS = ['lidar'] + ['CAMERA_%02d' % idx for idx in [1, 5, 6, 7, 8, 9]] 
# ddad_train = SynchronizedSceneDataset(
#     DDAD_TRAIN_VAL_JSON_PATH,
#     split='train',
#     datum_names=DATUMS,
#     generate_depth_from_datum='lidar'
# )
# print('Loaded DDAD train split containing {} samples'.format(len(ddad_train)))

class DDADDataset(data.Dataset):
    def __init__(self, num_scales, is_train, width=640, height=384):
        super(DDADDataset, self).__init__()
        
        split = 'train' if is_train == True else 'val'
        
        self.ddad_train_with_context = SynchronizedSceneDataset(
            DDAD_TRAIN_VAL_JSON_PATH,
            split=split,
            datum_names=('lidar', 'CAMERA_01',),
            generate_depth_from_datum='lidar',
            forward_context=1, 
            backward_context=1
        )
        
        self.num_scales = num_scales
        self.is_train = is_train
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        self.resize_tensor = {}
        self.width = width
        self.height = height
        self.interp = Image.ANTIALIAS
        
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
            self.resize_tensor[i] = transforms.Resize((self.height // s, self.width // s))


        self.to_tensor = transforms.ToTensor()
        
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.ddad_train_with_context)
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        
        samples = self.ddad_train_with_context[index] 
        front_cam_images = []
        for sample in samples:
            front_cam_images.append(sample[0]['rgb'])

        front_cam_images = [img.resize((self.width, self.height), Image.BILINEAR) for img in front_cam_images]
        inputs = {}

        do_color_aug = False # self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        inputs[("color", -1, -1)] = front_cam_images[0]
        inputs[("color", 0, -1)] = front_cam_images[1]
        # inputs[("color", 1, -1)] = front_cam_images[2]
        
        
        for scale in range(self.num_scales):
            K = np.zeros((4, 4), np.float32)
            K[:3, :3] = samples[1][0]['intrinsics'].copy()
            K[3][3] = 1
            
            K[0, :] *= self.width / 1936
            K[1, :] *= self.height / 1216 # self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        
        for i in [0, -1]:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            
        # print(inputs[("color", 0, 0)].shape)
        # Image.fromarray((inputs[("color_aug", 1, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_0.jpg')
        # Image.fromarray((inputs[("color_aug", 0, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_1.jpg')
        # Image.fromarray((inputs[("color_aug", -1, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_2.jpg')
        # # Image.fromarray((inputs[("color_aug", 1, 3)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_3.jpg')
        # exit(0)
        
        if not self.is_train:
            inputs[("depth")] = samples[1][0]['depth']
        
        return inputs

if __name__ == "__main__":
    # ddad_train = SynchronizedSceneDataset(
    #     DDAD_TRAIN_VAL_JSON_PATH,
    #     split='val',
    #     datum_names=DATUMS,
    #     generate_depth_from_datum='lidar',
    #     forward_context=1, 
    #     backward_context=1
    # )
    
    # ddad_train_with_context = SynchronizedSceneDataset(
    #     DDAD_TRAIN_VAL_JSON_PATH,
    #     split='train',
    #     datum_names=DATUMS, #('CAMERA_01',),
    #     generate_depth_from_datum='lidar',
    #     forward_context=1, 
    #     backward_context=1
    # )
    # print('Loaded DDAD train split containing {} samples'.format(len(ddad_train)))
    
    dataset = DDADDataset(4, is_train=False)
    from torch.utils.data.dataloader import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for item in loader:
        a = 10
