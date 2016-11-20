import glob

from scipy import misc

from constants import *


class Dataset:
    def __init__(self, src_dir, data_dir, im_size, name, num_train=None, target_size=None, num_classes=0):
        self.num_classes = num_classes
        self.src_data_dir = src_dir
        self.im_size = im_size
        if target_size:
            self.target_size = target_size
        else:
            self.target_size = self.im_size
        self.dims = self.target_size + (3,)
        if data_dir:
            self.data_dir = data_dir
            self.train_dir = os.path.join(self.data_dir, 'Train/')
            self.val_dir = os.path.join(self.data_dir, 'Validation/')
            # Check if dataset has already been preprocessed
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
                os.mkdir(self.train_dir)
                os.mkdir(self.val_dir)

                self.process_dataset()
            # Paths to data-files with randomized train and test images
            self.val_files = glob.glob('%s/image/%s*' % (self.val_dir, 'val'))
            self.train_files = glob.glob('%s/image/%s*' % (self.train_dir, 'train'))
            if num_train:
                self.train_files = self.train_files[:num_train]
            else:
                num_train = len(self.train_files)
            self.num_train = num_train
            self.num_val = len(self.val_files)
            self.name = '{}_numTrain_{}_targetSize_{}'.format(name, self.num_train, self.target_size)

    def process_dataset(self):
        # To be implemented by sub-classes
        pass

    def prepare_dirs_toon(self):
        os.mkdir(os.path.join(self.train_dir, 'cartoon/'))
        os.mkdir(os.path.join(self.train_dir, 'edge/'))
        os.mkdir(os.path.join(self.train_dir, 'image/'))
        os.mkdir(os.path.join(self.val_dir, 'cartoon/'))
        os.mkdir(os.path.join(self.val_dir, 'edge/'))
        os.mkdir(os.path.join(self.val_dir, 'image/'))


class ImagenetToon(Dataset):
    def __init__(self, num_train=None, target_size=None):
        Dataset.__init__(self, src_dir=IMAGENET_SRC_DIR, data_dir=IMAGENET_TOON_DATADIR, im_size=(256, 256),
                         name='Imagenet_Toon', num_train=num_train, target_size=target_size)

    def process_dataset(self):
        from prepare_data_jpeg import process_dataset
        self.prepare_dirs_toon()
        val_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_val/')
        train_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, self.im_size)
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, self.im_size)


class Imagenet(Dataset):
    def __init__(self, num_train=None, target_size=(227, 227), im_size=(256, 256)):
        Dataset.__init__(self, IMAGENET_SRC_DIR, None, im_size, name='Imagenet', num_train=num_train,
                         target_size=target_size, num_classes=1000)
        self.train_dir = IMAGENET_TRAIN_DIR
        self.val_dir = IMAGENET_VAL_DIR
        self.num_classes = 1000
        self.num_train = 1281167
        self.num_val = 50000
        self.name = 'Imagenet_numTrain_{}_targetSize_{}'.format(num_train, target_size)


class TinyImagenetToon(Dataset):
    def __init__(self, src_dir=TINYIMAGENET_SRC_DIR, data_dir=TINYIMAGENET_TOON_DATADIR, im_size=(64, 64),
                 name='TinyImagenet', num_train=None, target_size=(64, 64)):
        Dataset.__init__(self, src_dir, data_dir, im_size, name, num_train, target_size)

    def process_dataset(self):
        from prepare_data_jpeg import process_dataset
        self.prepare_dirs_toon()
        val_dir = os.path.join(self.src_data_dir, 'val/images/')
        train_dir = os.path.join(self.src_data_dir, 'train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, self.im_size)
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, self.im_size)


class TinyImagenet(Dataset):
    def __init__(self, src_dir=TINYIMAGENET_SRC_DIR, data_dir=None, im_size=(64, 64),
                 name='TinyImagenet', num_train=None, target_size=(64, 64)):
        Dataset.__init__(self, src_dir, data_dir, im_size, name, num_train, target_size)
        self.num_train = 100000
        self.num_val = 10000
        self.train_dir = os.path.join(TINYIMAGENET_SRC_DIR, 'train/')
        self.val_dir = os.path.join(TINYIMAGENET_SRC_DIR, 'val/images/')
        self.num_classes = 200


class CIFAR10_Toon(Dataset):
    def __init__(self, src_dir=None, data_dir=CIFAR10_TOON_DATADIR, im_size=(32, 32), name='CIFAR10Toon',
                 num_train=None,
                 target_size=None):
        Dataset.__init__(self, src_dir, data_dir, im_size, name, num_train, target_size)

    def process_dataset(self):
        self.prepare_dirs_toon()
        # Otherwise get the data set and process it
        from cartooning import cartoonify, auto_canny
        from keras.datasets import cifar10
        (img_train, _), (img_val, _) = cifar10.load_data()

        for i in range(len(img_train)):
            filename = 'train_{}'.format(i + 1)
            toon_path = os.path.join(self.train_dir, 'cartoon/{}.JPEG'.format(filename))
            edge_path = os.path.join(self.train_dir, 'edge/{}.JPEG'.format(filename))
            img_path = os.path.join(self.train_dir, 'image/{}.JPEG'.format(filename))
            image_cartoon = cartoonify(img_train[i], num_donw_samp=1)
            image_edge = auto_canny(img_train[i])
            misc.imsave(toon_path, image_cartoon, format='JPEG')
            misc.imsave(edge_path, image_edge, format='JPEG')
            misc.imsave(img_path, img_train[i], format='JPEG')
            if not i % 500:
                print('Processed %d of %d train-images.' % (i, len(img_train)))

        for i in range(len(img_val)):
            filename = 'val_{}'.format(i + 1)
            toon_path = os.path.join(self.val_dir, 'cartoon/{}.JPEG'.format(filename))
            edge_path = os.path.join(self.val_dir, 'edge/{}.JPEG'.format(filename))
            img_path = os.path.join(self.val_dir, 'image/{}.JPEG'.format(filename))
            image_cartoon = cartoonify(img_val[i], num_donw_samp=1)
            image_edge = auto_canny(img_val[i])
            misc.imsave(toon_path, image_cartoon, format='JPEG')
            misc.imsave(edge_path, image_edge, format='JPEG')
            misc.imsave(img_path, img_val[i], format='JPEG')
            if not i % 500:
                print('Processed %d of %d val-images.' % (i, len(img_val)))


class CIFAR10(Dataset):
    def __init__(self, src_dir=None, data_dir=CIFAR10_DATADIR, im_size=(32, 32), name='CIFAR10', num_train=50000,
                 target_size=None):
        Dataset.__init__(self, src_dir, data_dir, im_size, name, num_train, target_size, num_classes=10)
        self.num_val = 10000

    def process_dataset(self):
        import numpy as np
        from keras.datasets import cifar10
        (X_train, y_train), (X_val, y_val) = cifar10.load_data()
        labels = np.unique(y_train)
        for l in labels:
            os.mkdir(os.path.join(self.train_dir, 'class_{}/'.format(l)))
            os.mkdir(os.path.join(self.val_dir, 'class_{}/'.format(l)))

        for i in range(len(X_train)):
            filename = 'train_{}'.format(i + 1)
            im_path = os.path.join(self.train_dir, 'class_{}/{}.JPEG'.format(y_train[i][0], filename))
            misc.imsave(im_path, X_train[i], format='JPEG')
            if not i % 500:
                print('Processed %d of %d train-images.' % (i, len(X_train)))

        for i in range(len(X_val)):
            filename = 'val_{}'.format(i + 1)
            im_path = os.path.join(self.val_dir, 'class_{}/{}.JPEG'.format(y_val[i][0], filename))
            misc.imsave(im_path, X_val[i], format='JPEG')
            if not i % 500:
                print('Processed %d of %d train-images.' % (i, len(X_val)))
