import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfs
from PIL import Image
import scipy.io
import os
from enum import IntEnum

# format
# data['record']: {
#   'filename': string, filename
#   'folder': string
#   'source': struct for database etc
#   'imgname': string, path
#   'size': dimensions
#       'height'
#       'width'
#       'depth'
#   'segmented': 0/1 ?
#   'imgsize': dimensions [h,w,c]
#   'database': source database
#   'objects': list of objects
#       'class': string, class
#       'view': string, Frontal/Rear/Left/Right
#       'bbox': bounding box
#       'bndbox': bounding box as map (might only exist sometimes)
#       'orglabel': string includes a bunch of stuff (might only exist sometimes)
#       'truncated': 0/1
#       'occluded': 0/1
#       'difficult': 0/1
#       'anchors': list of anchors and their coordinates in image, contents depend on class, map between strings and anchors
#           'location': [], or [x,y] position in image
#           'status': something
#       'viewpoint': essentially angle
#           'azimuth_coarse':
#           'azimuth':
#           'elevation_coarse':
#           'elevation':
#           'distance':
#           'px': center
#           'py': center
#           'theta':
#           'error':
#           'interval_azimuth':
#           'interval_elevation':
#           'num_anchor':
#           'viewport':
#       'cad_index': related to which cad was used
#       'polygon': empty list
#       'point': empty list
#       'part': empty list
#       'hasparts': 0/1
#       'actions': list
#       'hasactions': 0/1
#       'mask': 0/1
#  sometimes additional dimensions of size 1 is inserted probably due to matlab


def get_mat_element(data):
    while isinstance(data, np.ndarray):
        if len(data) == 0:
            raise PascalParseError("Encountered Empty List")
        if len(data) > 1:
            x = data[0]
            for y in data:
                if y != x:
                    print(data[0])
                    print(data[1])
                    raise (Exception("blah" + str(data)))
        data = data[0]
    return data


def get_mat_list(data):
    data_old = data
    while len(data) == 1:
        data_old = data
        data = data[0]
    if isinstance(data, np.void):
        return data_old
    return data


def pascal3d_get_bbox(data):
    names = data.dtype.names
    if 'bbox' in names:
        bbox = data['bbox']
        bbox = get_mat_list(bbox)
        return list(map(float, bbox))
    elif 'bndbox' in names:
        raise Exception("NOT IMPLEMENTED")
    raise PascalParseError("could not parse bounding box")


class PascalParseError(Exception):
    def __init__(self, string):
        super().__init__(string)


class PascalClasses(IntEnum):
    AEROPLANE = 1
    BICYCLE = 2
    BOAT = 3
    BOTTLE = 4
    BUS = 5
    CAR = 6
    CHAIR = 7
    DININGTABLE = 8
    MOTORBIKE = 9
    SOFA = 10
    TRAIN = 11
    TVMONITOR = 12

    def __str__(self):
        return self.name.lower()


pascal_3d_str_enum_map = {}
for v in PascalClasses:
    pascal_3d_str_enum_map[str(v)] = v

failed_parse_strings = set()


def pascal3d_get_class(data):
    class_str = get_mat_element(data['class'])
    try:
        return pascal_3d_str_enum_map[class_str.lower()]
    except KeyError:
        failed_parse_strings.add(class_str.lower())
        # print("unknown class: " + class_str)
        raise PascalParseError("could not parse class")


def pascal3d_idx_to_str(idx):
    return str(PascalClasses(idx))


def parse_single_angle(viewpoint, angle_name):
    names = viewpoint.dtype.names
    if angle_name in names:
        try:
            angle = get_mat_element(viewpoint[angle_name])
            return float(angle)
        except PascalParseError:
            pass
    angle_name_coarse = angle_name + "_coarse"
    if angle_name_coarse in names:
        angle = get_mat_element(viewpoint[angle_name_coarse])
        return float(angle)
    raise PascalParseError("No angle found")

def pascal3d_get_angle(data):
    viewpoint = get_mat_element(data['viewpoint'])
    azimuth = parse_single_angle(viewpoint, 'azimuth')
    elevation = parse_single_angle(viewpoint, 'elevation')
    theta = parse_single_angle(viewpoint, 'theta')
    if azimuth == 0 and elevation == 0 and theta == 0:
        raise PascalParseError("Angle probably not entered")
    return [azimuth, elevation, theta]  # note in degree


def pascal3d_get_point(data):
    viewpoint = get_mat_element(data['viewpoint'])
    px = float(get_mat_element(viewpoint['px']))
    py = float(get_mat_element(viewpoint['py']))
    return [px, py]


def pascal3d_get_distance(data):
    viewpoint = get_mat_element(data['viewpoint'])
    return float(get_mat_element(viewpoint['distance']))


DICT_BOUNDING_BOX = 'bounding_box'
DICT_CLASS = 'class'
DICT_ANGLE = 'angle'
DICT_OCCLUDED = 'occluded'
DICT_TRUNCATED = 'truncated'
DICT_DIFFICULT = 'difficult'
DICT_POINT = 'px'
DICT_OBJECT_LIST = 'obj_list'
DICT_OBJECT_INSTANCE = 'obj_instance'
DICT_FILENAME = 'filename'
DICT_DISTANCE = 'distance'
DICT_CAMERA = 'camera'
DICT_CAD_INDEX = 'cad_index'


def get_pascal_camera_params(mat_data):
    viewpoint = get_mat_element(mat_data['viewpoint'])
    try:
        focal = get_mat_element(viewpoint['focal'])
    except PascalParseError:
        print("default_focal")
        focal = 1
    if focal != 1:
        print("focal {}".format(focal))
    try:
        viewport = get_mat_element(viewpoint['viewport'])
    except PascalParseError:
        print("default_viewpoer")
        viewport = 3000
    if viewport != 3000:
        print("viewport {}".format(viewport))
    return float(focal), float(viewport)


def mat_data_to_dict_data(mat_data, folder):
    record = get_mat_element(mat_data['record'])
    ret = {}
    objects = []
    mat_objects = get_mat_list(record['objects'])
    for obj in mat_objects:
        ret_obj = {}
        try:
            ret_obj[DICT_BOUNDING_BOX] = pascal3d_get_bbox(obj)
            ret_obj[DICT_CLASS] = pascal3d_get_class(obj).value
            ret_obj[DICT_ANGLE] = pascal3d_get_angle(obj)
            ret_obj[DICT_OCCLUDED] = bool(get_mat_element(obj['occluded']))
            ret_obj[DICT_TRUNCATED] = bool(get_mat_element(obj['truncated']))
            ret_obj[DICT_POINT] = pascal3d_get_point(obj)
            ret_obj[DICT_DIFFICULT] = bool(get_mat_element(obj['difficult']))
            ret_obj[DICT_DISTANCE] = pascal3d_get_distance(obj)
            ret_obj[DICT_CAMERA] = get_pascal_camera_params(obj)

            ret_obj[DICT_CAD_INDEX] = int(get_mat_element(obj['cad_index']))
            objects.append(ret_obj)
        except PascalParseError as e:
            pass
    ret[DICT_OBJECT_LIST] = objects
    ret[DICT_FILENAME] = os.path.join(folder, get_mat_element(record['filename']))
    return ret

def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix

def process_annotated_image(
    im, left, top, right, bottom, azimuth, elevation, theta,
    augment, reverse_theta, crop):

    # perturb bbox randomly
    # inputs are matlab (start at 1)
    if augment:
        max_shift = 7
        left = left - 1 + np.random.randint(-max_shift, max_shift+1)
        top = top - 1 + np.random.randint(-max_shift, max_shift+1)
        right = right - 1 + np.random.randint(-max_shift, max_shift+1)
        bottom = bottom - 1 + np.random.randint(-max_shift, max_shift+1)
    else:
        left = left - 1
        top = top - 1
        right = right - 1
        bottom = bottom - 1
    width, height = im.size
    left = min(max(left,0), width)
    top = min(max(top, 0), height)
    right = min(max(right, 0), width)
    bottom = min(max(bottom, 0), height)
    # Resizing can change aspect ratio, so we could adjust the ground truth
    # rotation accordingly (leaving as-is since initial results didn't change)
    if crop:
        im = im.crop((left, top, right, bottom))
    im = im.resize([224,224])

    # Inputs are in degrees, convert to rad.
    az = azimuth* np.pi / 180.0
    el = elevation * np.pi / 180.0
    th = theta * np.pi / 180.0
    # Reversing theta for RenderForCNN data since that theta was set from filename
    # which has negative theta (see github.com/ShapeNet/RenderForCNN).
    if reverse_theta:
        th = -th

    if augment:
        # Flip
        rand = np.random.uniform(0,1)
        if rand < 0.5:
            az = -az
            th = -th
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im = tfs.ColorJitter(brightness=0.1, contrast=0.5, hue=0.2, saturation=0.5)(im)
        #im_crop = tf.clip_by_value(im_crop, 0.0, 1.0)

    # R = R_z(th) * R_x(el−pi/2) * R_z(−az)
    R1 = compute_rotation_matrix_from_euler(torch.Tensor([0,0,-az]).unsqueeze(0))
    R2 = compute_rotation_matrix_from_euler(torch.Tensor([el-np.pi/2.0, 0, th]).unsqueeze(0))
    R = torch.bmm(R2, R1)

    return tfs.ToTensor()(im), R

class PascalParseError(Exception):
    def __init__(self, string):
        super().__init__(string)

def get_mat_element(data):
    while isinstance(data, np.ndarray):
        if len(data) == 0:
            raise PascalParseError("Encountered Empty List")
        if len(data) > 1:
            x = data[0]
            for y in data:
                if y != x:
                    print(data[0])
                    print(data[1])
                    raise(Exception("blah" + str(data)))
        data = data[0]
    return data


def create_imagenet_anno(data_folder, save_folder, category, validation_split_size=0.3):
    ImageNet_anno_folder = os.path.join(data_folder, 'Annotations', category + '_imagenet')
    ImageNet_img_folder = os.path.join(data_folder, 'Images', category + '_imagenet')
    imagenet_split_train_path = os.path.join(data_folder, 'Image_sets', category + '_imagenet_train.txt')
    imagenet_split_test_path = os.path.join(data_folder, 'Image_sets', category + '_imagenet_val.txt')
    # train and val
    train_val_split = []
    with open(imagenet_split_train_path,'r') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            while l[-1] in ('\n', '\r'):
                l = l[:-1]
                if len(l) == 0:
                    continue
            train_val_split.append(l)
    train_val_split = sorted(train_val_split)
    val_idx = (np.arange(len(train_val_split) * validation_split_size) / validation_split_size).astype(np.int)
    val_split = [train_val_split[i] for i in val_idx]
    train_split = sorted(list(set(train_val_split) - set(val_split)))
    # test
    test_split = []
    with open(imagenet_split_test_path, 'r') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            while l[-1] in ('\n', '\r'):
                l = l[:-1]
                if len(l) == 0:
                    continue
            test_split.append(l)

    split = []
    split.append(train_split)
    split.append(val_split)
    split.append(test_split)
    save_path = []
    save_path.append(os.path.join(save_folder, category+'_imagenet_train'))
    save_path.append(os.path.join(save_folder, category + '_imagenet_val'))
    save_path.append(os.path.join(save_folder, category + '_imagenet_test'))
    name_lst = ['train', 'val', 'test']
    #create new annotation of ImageNet
    for i in range(len(name_lst)):
        if not os.path.isdir(save_path[i]):
            os.mkdir(save_path[i])
        for instance in split[i]:
            annopath = os.path.join(ImageNet_anno_folder, instance+'.mat')
            anno = scipy.io.loadmat(annopath)
            dict = mat_data_to_dict_data(anno, ImageNet_img_folder)
            for num, obj in enumerate(dict['obj_list']):
                obj['imgpath'] = dict['filename']
                if obj['occluded'] or obj['truncated'] or obj['difficult']:
                    continue
                save_file = os.path.join(save_path[i], instance+'_'+str(num)+'.npy')
                np.save(save_file, obj)
        print('Total imagenet %s obj number for %s: %d' %(category, name_lst[i], len(os.listdir(save_path[i]))))

def create_pascal_anno(data_folder, save_folder, category):
    pascal_anno_folder = os.path.join(data_folder, 'Annotations', category + '_pascal')
    pascal_img_folder = os.path.join(data_folder, 'Images', category + '_pascal')
    # train
    train_split = os.listdir(pascal_anno_folder)
    save_path = os.path.join(save_folder, category+'_pascal_train')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for instance in train_split:
        annopath = os.path.join(pascal_anno_folder, instance)
        anno = scipy.io.loadmat(annopath)
        dict = mat_data_to_dict_data(anno, pascal_img_folder)
        for num, obj in enumerate(dict['obj_list']):
            obj['imgpath'] = dict['filename']
            if obj['occluded'] or obj['truncated'] or obj['difficult']:
                continue
            save_file = os.path.join(save_path, instance[:-4] + '_' + str(num) + '.npy')
            np.save(save_file, obj)
    print('Total pascal %s obj number: %d' %(category, len(os.listdir(save_path))))


class Pascal3dDataset(torch.utils.data.Dataset):
    def __init__(self, anno_folder, augment=False, voc_train_addition_folder=None):
        self.anno_paths = [os.path.join(anno_folder, i) for i in os.listdir(anno_folder)]
        if voc_train_addition_folder != None:
            self.anno_paths.extend(os.listdir(voc_train_addition_folder))
        self.augment = augment
        self.size = len(self.anno_paths)
        print('Load Pascal Dataset, Length:', self.size)

    def __getitem__(self, idx):
        annopath = self.anno_paths[idx]
        anno = np.load(annopath, allow_pickle=True).item()
        imgpath = anno['imgpath']
        img = Image.open(imgpath)
        left, top, right, bottom = anno['bounding_box']
        azimuth, elevation, theta = anno['angle']
        img, R = process_annotated_image(img, left, top, right, bottom, azimuth, elevation, theta, augment=self.augment, reverse_theta=False, crop=True)
        return img, R

    def __len__(self):
        return  self.size


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, syn_folder, category_id):
        img_folder = os.path.join(syn_folder, category_id)
        instance_lst = os.listdir(img_folder)
        self.paths = []
        for i in instance_lst:
            path_i = os.path.join(img_folder, i)
            lst_i = os.listdir(path_i)
            for j in lst_i:
                if '.png' in j:
                    self.paths.append(os.path.join(path_i, j))
        self.size = len(self.paths)
        print('Load Synthetic Dataset, Length:', self.size)

    def __getitem__(self, idx):
        fpath = self.paths[idx]
        img = Image.open(fpath)
        anno = fpath.split('_')
        azimuth = int(anno[-4][1:])
        elevation = int(anno[-3][1:])
        theta = int(anno[-2][1:])

        img, R = process_annotated_image(img, 0, 0, 0, 0, azimuth, elevation, theta, augment=False,
                                         reverse_theta=True, crop=False)
        return img, R

    def __len__(self):
        return self.size


name2id = {
    'aeroplane':'02691156',
    'bicycle':'02834778',
    'boat':'02858304',
    'bottle':'02876657',
    'bus':'02924116',
    'car':'02958343',
    'chair':'03001627',
    'diningtable':'04379243',
    'motorbike':'03790512',
    'sofa': '04256520',
    'train':'04468005',
    'tvmonitor':'03211117',
}

def get_dataloader(mode, config):
    pascal3d_path = config.pascal3d_path
    syn_path = config.syn_path
    save_anno_path = os.path.join(config.pascal3d_path, 'my_anno')
    category = config.category
    if config.create_anno and mode == 'train':
        if not os.path.isdir(save_anno_path):
            os.mkdir(save_anno_path)
        create_imagenet_anno(pascal3d_path, save_anno_path, category)
        create_pascal_anno(pascal3d_path, save_anno_path, category)
    if mode == 'train':
        anno_path_train = os.path.join(save_anno_path, category + '_imagenet_train')
        if config.voc_train:
            anno_path_train_voc = os.path.join(save_anno_path, category + '_pascal_train')
            pascal3d_train_dataset = Pascal3dDataset(anno_path_train, augment=True,voc_train_addition_folder=anno_path_train_voc)
        else:
            pascal3d_train_dataset = Pascal3dDataset(anno_path_train, augment=True)
        train_loader1 = DataLoader(pascal3d_train_dataset, batch_size=int(config.batch_size/2), shuffle=True, num_workers=config.num_workers)
        syn_train_dataset = SyntheticDataset(syn_path, name2id[category])
        train_loader2 = DataLoader(syn_train_dataset, batch_size=int(config.batch_size/2), shuffle=True, num_workers=config.num_workers)
        return train_loader1, train_loader2
    elif mode == 'test':
        anno_path_test = os.path.join(save_anno_path, category + '_imagenet_test')
        pascal3d_test_dataset = Pascal3dDataset(anno_path_test)
        test_loader = DataLoader(pascal3d_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return test_loader

if __name__ == '__main__':
    import argparse
    from os.path import join 
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_root", type=str, default='../dataset')
    args = arg_parser.parse_args()

    pascal3d_path = join(args.data_root, 'PASCAL3D+_release1.1')
    syn_path = join(args.data_root, 'syn_images_cropped_bkg_overlaid')
    save_anno_path = os.path.join(pascal3d_path, 'my_anno')
    category_lst = ['aeroplane','sofa', 'bicycle','boat','bottle','bus',
    'car', 'chair', 'diningtable', 'motorbike', 'train', 'tvmonitor']

    if not os.path.isdir(save_anno_path):
        os.mkdir(save_anno_path)

    #create annotation first!!!
    for category in category_lst:
        create_imagenet_anno(pascal3d_path, save_anno_path, category)
        # # No need to do below line
        # create_pascal_anno(pascal3d_path, save_anno_path, category)
    '''
    anno_path_train = os.path.join(save_anno_path, category+'_imagenet_train')
    anno_path_val = os.path.join(save_anno_path, category+'_imagenet_val')
    anno_path_test = os.path.join(save_anno_path, category+'_imagenet_test')
    pascal3d_train_dataset = Pascal3dDataset(anno_path_train, augment=True)
    pascal3d_test_dataset = Pascal3dDataset(anno_path_test)
    pascal3d_val_dataset = Pascal3dDataset(anno_path_val)

    syn_train_dataset = SyntheticDataset(syn_path, name2id[category])
    train_loader1 = DataLoader(pascal3d_train_dataset, batch_size=16, shuffle=True, num_workers=4)
    train_loader2 = DataLoader(syn_train_dataset, batch_size=16, shuffle=True, num_workers=4)

    #training
    for real_data, syn_data in zip(train_loader1,train_loader2):
        real_img, real_gt = real_data
        syn_img, syn_gt = syn_data
        train_data = torch.cat((real_img,syn_img),0) #[32,3,224,224]
        train_gt = torch.cat((real_gt,syn_gt),0)    #[32,1,3,3]
        print(train_data.shape, train_gt.shape)
    '''