# *_*coding:utf-8 *_*
import time, warnings, os, importlib, json, torch, pickle, numpy as np, point_cloud_utils as pcu
from torch.utils.data import Dataset
from scipy.special import softmax
from tqdm import tqdm

warnings.filterwarnings('ignore')

n_class, n_part = 16, 50
n_pc_test = 2874
n_xyz_max = 2947
ds_list = ['random', 'farthest', 'poisson']
n_point_list = [2048, 1024, 512, 256, 128]
epoch_list = list(range(300))
mapping_list = ['nn', 'uniform', 'linear', 'cosine', 'gaussian', 'exponential']
key_list = ['first', 'left', 'total']
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
cls_to_cat = list(seg_classes.keys())
cls_to_cat.sort()  # ['Airplane', 'Bag', 'Cap', ..., 'Table']

best_epoch_list = [['farthest,128', 63, 'DSfarthest-point_NP0128_EP300/checkpoints/ep062_model.pth'],
                   ['farthest,256', 67, 'DSfarthest-point_NP0256_EP300/checkpoints/ep066_model.pth'],
                   ['farthest,512', 88, 'DSfarthest-point_NP0512_EP300/checkpoints/ep087_model.pth'],
                   ['farthest,1024', 93, 'DSfarthest-point_NP1024_EP300/checkpoints/ep092_model.pth'],
                   ['farthest,2048', 87, 'DSfarthest-point_NP2048_EP300/checkpoints/ep086_model.pth'],
                   ['poisson,128', 107, 'DSpoisson-disk_NP0128_EP300/checkpoints/ep106_model.pth'],
                   ['poisson,256', 85, 'DSpoisson-disk_NP0256_EP300/checkpoints/ep084_model.pth'],
                   ['poisson,512', 69, 'DSpoisson-disk_NP0512_EP300/checkpoints/ep068_model.pth'],
                   ['poisson,1024', 68, 'DSpoisson-disk_NP1024_EP300/checkpoints/ep067_model.pth'],
                   ['poisson,2048', 71, 'DSpoisson-disk_NP2048_EP300/checkpoints/ep070_model.pth'],
                   ['random,128', 109, 'DSrandom_NP0128_EP300/checkpoints/ep108_model.pth'],
                   ['random,256', 89, 'DSrandom_NP0256_EP300/checkpoints/ep088_model.pth'],
                   ['random,512', 126, 'DSrandom_NP0512_EP300/checkpoints/ep125_model.pth'],
                   ['random,1024', 116, 'DSrandom_NP1024_EP300/checkpoints/ep115_model.pth'],
                   ['random,2048', 80, 'DSrandom_NP2048_EP300/checkpoints/ep079_model.pth']]


def get_best_epoch(case):
    for best_epoch in best_epoch_list:
        if best_epoch[0] == case:
            best_epoch, pretrained_model_file = best_epoch[1:]
            return best_epoch, pretrained_model_file
    print('ERROR:', case)
    exit()




def get_case(args, epoch=100, ds='random', n_point=2048):
    import sys
    sys.path.append("/home/lecun/Workspace/jun/Hogeony_SUPCS/paper_Sensors/models")
    MODEL = importlib.import_module('pointnet2_part_seg_msg')
    classifier = MODEL.get_model(n_class, n_part, normal_channel=False)

    if ds == 'random':
        case = 'DSrandom_NP%04d_EP300' % n_point
    elif ds == 'farthest':
        case = 'DSfarthest-point_NP%04d_EP300' % n_point
    elif ds == 'poisson':
        case = 'DSpoisson-disk_NP%04d_EP300' % n_point

    checkpoint = torch.load(os.path.join(args.dir_trained_model, case, 'checkpoints', 'ep%03d_model.pth' % epoch))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    return classifier


def get_metrics_empty(key_list=['first', 'left', 'total']):
    metrics = {}
    for key in key_list:
        metrics[key] = {'seen': 0,
                        'correct': 0,
                        'class_seen': [0 for _ in range(n_part)],
                        'class_correct': [0 for _ in range(n_part)],
                        'class_ious': {cat: [] for cat in seg_classes.keys()}}
    return metrics


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def prob_R(mapping, R, r_cut):
    if mapping == 'uniform':
        return np.array(list(map(lambda r: 1 if r <= r_cut else 0, R)), dtype=np.float32)
    elif mapping == 'linear':
        return np.array(list(map(lambda r: -r / r_cut + 1 if r <= r_cut else 0, R)))
    elif mapping == 'cosine':
        return np.array(list(map(lambda r: np.cos(r / r_cut * np.pi / 2) if r <= r_cut else 0, R)))
    elif mapping == 'gaussian':
        return np.exp(-R * R / (0.18 * r_cut * r_cut))
    elif mapping == 'exponential':
        return np.exp(-5.5 / r_cut * R)
    else:
        raise 'ERROR: not proper prob_func'


def prob_uniform(R, r_cut):
    return np.array(list(map(lambda r: 1 if r <= r_cut else 0, R)))


def prob_linear(R, r_cut):
    return np.array(list(map(lambda r: -r / r_cut + 1 if r <= r_cut else 0, R)))


def prob_cosine(R, r_cut):
    return np.array(list(map(lambda r: np.cos(r / r_cut * np.pi / 2) if r <= r_cut else 0, R)))


def prob_gaussian(R, r_cut):
    # np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return np.exp(-R * R / (0.18 * r_cut * r_cut))


def prob_exponential(R, r_cut):
    return np.exp(-5.5 / r_cut * R)


def get_seg_label_to_cat(mode='class'):
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for i, label in enumerate(seg_classes[cat]):
            if mode == 'class':
                seg_label_to_cat[label] = cat
            elif mode == 'part':
                seg_label_to_cat[label] = '%s_%d' % (cat, i)
            else:
                raise 'ERROR: mode'
    return seg_label_to_cat


def get_dist_mean(full_xyz):
    dist_list = []
    for target_xyz in full_xyz:
        dist = np.linalg.norm(full_xyz - target_xyz, ord=2, axis=1)
        dist_list.append(dist[dist > 0].min())
    return sum(dist_list) / len(dist_list)


def downsampling(ds: str, n_point: int, xyz: np.ndarray):
    if len(xyz) > n_point:
        if ds == 'poisson':
            idx = pcu.downsample_point_cloud_poisson_disk(xyz, num_samples=n_point)[:n_point]
            if len(idx) != n_point:
                idx = np.r_[idx, np.setdiff1d(np.arange(len(xyz)), idx)[:n_point - len(idx)]]
        elif ds == 'random':
            idx = np.random.choice(range(xyz.shape[0]), n_point, replace=False)
        elif ds == 'farthest':
            N, D = xyz.shape
            idx = np.zeros((n_point,), dtype=np.int32)
            distance = np.ones((N,)) * 1e10
            farthest = np.random.randint(0, N)
            for i in range(n_point):
                idx[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = np.argmax(distance, -1)
        else:
            assert 'Method is not proper'
    else:
        idx = np.arange(len(xyz))

    xyz_sampled = xyz[idx]

    return xyz_sampled.astype(np.float32), idx.astype(np.int32)


class ShapeNetSegDataset(Dataset):
    def __init__(self, root='./datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='train'):
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')

        path_testdataset = os.path.join(root, 'dataset_test.pickle')
        if split == 'test' and os.path.exists(path_testdataset):
            with open(path_testdataset, 'rb') as f:
                self.cache = pickle.load(f)
            print('Dataset Load:', path_testdataset)
        else:
            self.cat = {}

            with open(self.catfile, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]
            self.cat = {k: v for k, v in self.cat.items()}
            self.classes_original = dict(zip(self.cat, range(len(self.cat))))

            self.meta = {}
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
                test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            for item in self.cat:
                # print('category', item)
                self.meta[item] = []
                dir_point = os.path.join(self.root, self.cat[item])
                fns = sorted(os.listdir(dir_point))
                # print(fns[0][0:-4])
                if split == 'trainval':
                    fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
                elif split == 'train':
                    fns = [fn for fn in fns if fn[0:-4] in train_ids]
                elif split == 'val':
                    fns = [fn for fn in fns if fn[0:-4] in val_ids]
                elif split == 'test':
                    fns = [fn for fn in fns if fn[0:-4] in test_ids]
                else:
                    print('Unknown split: %s. Exiting..' % (split))
                    exit(-1)

                for fn in fns:
                    token = (os.path.splitext(os.path.basename(fn))[0])
                    self.meta[item].append(os.path.join(dir_point, token + '.txt'))

            self.datapath = []
            for item in self.cat:
                for fn in self.meta[item]:
                    self.datapath.append((item, fn))

            self.classes = {}
            for i in self.cat.keys():
                self.classes[i] = self.classes_original[i]

            self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                                'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                                'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                                'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

            self.cache = dict()
            for index in tqdm(range(len(self.datapath))):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                xyz = data[:, :3]
                seg = data[:, -1].astype(np.int32)
                n_xyz = len(xyz)
                if n_xyz == n_xyz_max:
                    self.cache[index] = (xyz, cls, seg, n_xyz)
                else:
                    self.cache[index] = (np.r_[xyz, np.zeros((n_xyz_max - n_xyz, 3)).astype(np.float32)],
                                         cls,
                                         np.r_[seg, np.zeros((n_xyz_max - n_xyz,)).astype(np.float32)],
                                         n_xyz)
            with open(path_testdataset, 'wb') as f:
                pickle.dump(self.cache, f)
            print('SAVE:', path_testdataset)

    def __getitem__(self, index):

        return self.cache[index]

    def __len__(self):
        return len(self.cache)