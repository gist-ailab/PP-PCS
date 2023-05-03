import os, sys
import argparse

from tools import *
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser("pp")
    parser.add_argument('--dir_dataset', help="dataset (ShapeNet) path")
    parser.add_argument('--dir_trained_model', help="pretrained deep learning model path")
    parser.add_argument('--save_path', help="root for save part-segmentation results")
    parser.add_argument('--base_root', help="base root of pp code")

    parser.add_argument('--d_cutoff_start', help="set a start point of d cut off")
    parser.add_argument('--d_cutoff_end', help="set a end point of d cut off")

    return parser.parse_args()


def get_softmax_probability_first(args, dataset_test):
    ds_list = ['random', 'farthest', 'poisson']
    n_point_list = [2048, 1024, 512, 256, 128]

    bs, epoch, ds, n_point = 100, -1, 'random', 2048
    device = torch.device('cuda:0')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False)

    path_results = os.path.join(args.dir_dataset, 'softmax_pred_first_info.txt')
    if not os.path.isfile(path_results):
        with open(path_results, 'w') as f:
            f.write('')
    with open(path_results, 'r') as f:
        lines = ['_'.join(line.split('\t')[:4]) for line in f.readlines()]

    for ds in ds_list:
        for n_point in n_point_list:
            if epoch == -1:
                best_epoch, _ = get_best_epoch(case='%s,%d' % (ds, n_point))
            else:
                best_epoch = epoch

            case = '%d_%s_%d_%d' % (bs, ds, n_point, best_epoch)
            if case in lines:
                print('EXIST: %s' % case)
                continue

            classifier = get_case(args, best_epoch - 1, ds, n_point).to(device)
            classifier.eval()

            idx_first = np.load(os.path.join(args.dir_dataset, 'idx_first_%s,%d.npy' % (ds, n_point)))

            time_start = time.time()
            for i_bs, (xyz_full, label_full, seg_truth, n_xyz) in enumerate(tqdm(loader_test)):
                i_sampled = idx_first[i_bs * bs:(i_bs + 1) * bs]
                xyz_sampled = torch.gather(xyz_full, 1, torch.tensor(np.stack([i_sampled, i_sampled, i_sampled]).transpose(1, 2, 0), dtype=torch.int64)).transpose(2, 1).to(device)
    
                with torch.no_grad():
                    pred_sampled, _ = classifier(xyz_sampled, to_categorical(label_full, n_class).to(device))
    
                if i_bs == 0:
                    pred_first = pred_sampled
                else:
                    pred_first = torch.vstack((pred_first, pred_sampled))

            pred_first = pred_first.softmax(dim=-1).cpu().numpy()
            time_elapsed = time.time() - time_start
            file_path = os.path.join(args.dir_dataset, 'softmax_pred_first_%s,%d,%d.npy' % (ds, n_point, best_epoch))
            np.save(file_path, pred_first)
            print('SAVE:', file_path)

            with open(path_results, 'a') as f:
                msg = '%d\t%s\t%d\t%d\t%f' % (bs, ds, n_point, best_epoch, time_elapsed)
                f.write(msg + '\n')
                print(msg)


def copy_pretrained_model():
    import os, shutil
    root = 'paper_Sensors'

    for case, best_epoch, pretrained_model_path in best_epoch_list:
        target_dir = os.path.join(root, 'pretrained_models', '/'.join(pretrained_model_path.split('/')[:-1]))
        os.makedirs(target_dir, exist_ok=True)

        source_path = os.path.join(args.dir_trained_model, pretrained_model_path)
        target_path = os.path.join(root, 'pretrained_models', pretrained_model_path)

        shutil.copy(source_path, target_path)
        print(source_path, '===>', target_path)


def pcs_softmax(options):
    path_results = os.path.join(args.save_results, 'results_softmax_epBest.txt')
    epoch = -1
    r_cut_base = 0

    n_point_list = [2048, 1024, 512, 256, 128]
    if options[:2] == 'nn':
        mapping_list = ['nn']
        ds = options.split('_')[1]
        ds_list = [ds]

        if 'poisson' in ds:
            device = torch.device('cuda:0')
            if '1' in ds:
                n_point_list = [512, 256]
            else:
                n_point_list = [2048, 1024, 128]
            ds_list = ['poisson']
        else:
            device = torch.device('cuda:1')
    else:
        ds_list = ['farthest', 'random', 'poisson']
        
        mapping_list, r_cut_base = options.split('_')
        mapping_list, r_cut_base = mapping_list.split(','), int(r_cut_base)

    for n_point in n_point_list:
        for ds in ds_list:
            if epoch == -1:
                best_epoch, _ = get_best_epoch(case='%s,%d' % (ds, n_point))
            else:
                best_epoch = epoch

            i_first = np.load(os.path.join(args.dir_dataset, 'idx_first_%s,%d.npy' % (ds, n_point)))
            pred_first = np.load(os.path.join(args.dir_dataset, 'softmax_pred_first_%s,%d,%d.npy' % (ds, n_point, best_epoch)))

            for mapping in mapping_list:
                with open(path_results, 'r') as f:
                    case_list = ['_'.join(x.split('\t')[:5]) for x in f.readlines()]
                case = '%d_%s_%d_%s,%d' % (best_epoch, ds, n_point, mapping, r_cut_base)
                # print(case_list[0], case)
                if case in case_list:
                    print('EXIST:', case)
                    continue
                pred_full = np.zeros((n_pc_test, n_xyz_max, n_part))
                metrics = get_metrics_empty()
                i_mapping_list = []

                time_start = time.time()

                for i_pc in tqdm(range(len(dataset_test.cache))):
                    xyz_full, label_full, seg_truth, n_xyz = dataset_test.cache[i_pc]
                    label_full, seg_truth = torch.tensor(label_full, dtype=torch.int32), seg_truth.astype(int)

                    pred_full[i_pc, i_first[i_pc]] = pred_first[i_pc]
                    i_labeled = i_first[i_pc]
                    i_unlabeled = np.setdiff1d(np.arange(n_xyz), i_labeled)
                    i_mapping = 0
                    if 'nn' != mapping:
                        r_cut = r_cut_base / 100
                        r_max = np.linalg.norm(xyz_full, ord=2, axis=1).max()

                        while len(i_unlabeled) > 0:
                            
                            i_neighbor_mapped = np.array([], dtype=np.int)
                            for i_source in i_labeled:
                                r_unlabeled = np.linalg.norm(xyz_full[i_unlabeled] - xyz_full[i_source], ord=2, axis=1) / r_max
                                i = np.where(r_unlabeled <= r_cut)[0]

                                if len(i) > 0:
                                    i_neighbor = i_unlabeled[i]
                                    pred_full[i_pc, i_neighbor] += np.matmul(prob_R(mapping, r_unlabeled[i], r_cut)[:, None], pred_full[i_pc][i_source][None, :])
                                    i_neighbor_mapped = np.r_[i_neighbor_mapped, i_neighbor]
                            if len(i_neighbor_mapped) > 0:
                                pred_full[i_pc, i_neighbor_mapped] = softmax(pred_full[i_pc, i_neighbor_mapped], axis=-1)
                                i_unlabeled = np.setdiff1d(i_unlabeled, i_neighbor_mapped)
                                i_labeled = np.setdiff1d(np.arange(n_xyz), i_unlabeled)
                                
                            else:
                                r_cut *= 1.5

                            i_mapping += 1
                    else:
                        classifier = get_case(args, best_epoch - 1, ds, n_point).to(device)

                        classifier.eval()
                        while len(i_unlabeled) > 0:
                
                            if len(i_unlabeled) > n_point:
                                _, i = downsampling(ds, n_point, xyz_full[i_unlabeled])
                                i_sampled = i_unlabeled[i]
                                with torch.no_grad():
                                    pred_sampled, _ = classifier(torch.tensor(xyz_full[i_sampled]).unsqueeze(0).transpose(2, 1).to(device), to_categorical(label_full, n_class).to(device))
                                pred_full[i_pc][i_sampled] = pred_sampled[0].softmax(dim=-1).cpu().numpy()
                                i_unlabeled = np.setdiff1d(i_unlabeled, i_sampled)

                            elif len(i_unlabeled) < n_point:
                                i_labeled = np.setdiff1d(np.arange(n_xyz), i_unlabeled)
                                _, i = downsampling(ds, n_point, xyz_full[i_labeled])
                                i_sampled = np.r_[i_unlabeled, i_labeled[i[:n_point - len(i_unlabeled)]]]
                                with torch.no_grad():
                                    pred_sampled, _ = classifier(torch.tensor(xyz_full[i_sampled]).unsqueeze(0).transpose(2, 1).to(device), to_categorical(label_full, n_class).to(device))
                                pred_full[i_pc][i_unlabeled] = pred_sampled[0][:len(i_unlabeled)].softmax(dim=-1).cpu().numpy()
                                i_unlabeled = np.array([])

                            else:
                                i_sampled = i_unlabeled
                                with torch.no_grad():
                                    pred_sampled, _ = classifier(torch.tensor(xyz_full[i_sampled]).unsqueeze(0).transpose(2, 1).to(device), to_categorical(label_full, n_class).to(device))
                                pred_full[i_pc][i_sampled] = pred_sampled[0].softmax(dim=-1).cpu().numpy()
                                i_unlabeled = np.array([])
                            i_mapping += 1
                    i_mapping_list.append(i_mapping)
                    
                    """evaluation pp method for part-segmentation"""
                    cat = cls_to_cat[label_full[0]]
                    for key, i_metrics in zip(key_list, (i_first[i_pc],
                                                         np.delete(np.arange(n_xyz), i_first[i_pc]),
                                                         np.arange(n_xyz))):
                        key_seg_truth = seg_truth[i_metrics]
                        key_seg_pred = pred_full[i_pc][i_metrics]
                        key_seg_pred_label = key_seg_pred.argmax(-1)
                        key_n_correct = np.sum(key_seg_pred_label == key_seg_truth)

                        metrics[key]['seen'] += len(i_metrics)
                        metrics[key]['correct'] += key_n_correct

                        metrics[key]['class_seen'][label_full[0]] += len(i_metrics)
                        metrics[key]['class_correct'][label_full[0]] += key_n_correct
                        metrics[key]['part_ious'] = [0.0 for _ in range(len(seg_classes[cat]))]
                        for i_part in seg_classes[cat]:
                            if (np.sum(key_seg_truth == i_part) == 0) and (np.sum(key_seg_pred_label == i_part) == 0):  # part is not present, no prediction as well
                                metrics[key]['part_ious'][i_part - seg_classes[cat][0]] = 1.0
                            else:
                                metrics[key]['part_ious'][i_part - seg_classes[cat][0]] = np.sum((key_seg_truth == i_part) & (key_seg_pred_label == i_part)) / float(np.sum((key_seg_truth == i_part) | (key_seg_pred_label == i_part)))
                        metrics[key]['class_ious'][cat].append(np.nanmean(metrics[key]['part_ious']))
                time_lapse = time.time() - time_start

                for key in key_list:
                    metrics[key]['class_ious_mean'] = metrics[key]['class_ious'].copy()
                    metrics[key]['instance_ious_mean'] = []
                    for cat in metrics[key]['class_ious'].keys():
                        for iou in metrics[key]['class_ious'][cat]:
                            metrics[key]['instance_ious_mean'].append(iou)
                        metrics[key]['class_ious_mean'][cat] = np.nanmean(metrics[key]['class_ious'][cat])
                    metrics[key]['n_point'] = metrics[key]['seen']
                    metrics[key]['accuracy'] = metrics[key]['correct'] / float(metrics[key]['seen'])
                    metrics[key]['class_avg_accuracy'] = np.nanmean(np.array(metrics[key]['class_correct']) / np.array(metrics[key]['class_seen'], dtype=np.float))
                    metrics[key]['class_avg_iou'] = np.nanmean(list(metrics[key]['class_ious_mean'].values()))
                    metrics[key]['instance_avg_iou'] = np.nanmean(metrics[key]['instance_ious_mean'])
                metrics['epoch'] = best_epoch
                metrics['downsampling'] = ds
                metrics['n_point'] = n_point
                metrics['mapping'] = 'nn,0' if mapping == 'nn' else '%s,%d' % (mapping, r_cut_base)
                metrics['i_mapping'] = np.array(i_mapping_list)
                metrics['time_lapse'] = time_lapse

                pkl_path = os.path.join(args.dir_dataset, 'softmax_metrics_%s,%d,%d,%s.pickle' % (ds, n_point, metrics['epoch'], metrics['mapping']))
                with open(pkl_path, 'wb') as f:
                    pickle.dump(metrics, f)
                print('SAVE:', pkl_path)

                with open(path_results, 'a') as f:
                    msg = '%d\t%s\t%d\t%s\t%.3f\t%f' % (metrics['epoch'], metrics['downsampling'], metrics['n_point'], metrics['mapping'], metrics['i_mapping'].mean(), metrics['time_lapse'])
                    for key in key_list:
                        msg += '\t%.3f\t%.3f\t%.3f\t%.3f' % (metrics[key]['accuracy'] * 100, metrics[key]['class_avg_accuracy'] * 100, metrics[key]['class_avg_iou'] * 100, metrics[key]['instance_avg_iou'] * 100)
                    for key in key_list:
                        msg += '\t%d' % metrics[key]['n_point']
                    f.write(msg + '\n')
                    print(msg)


if __name__ == '__main__':
    import sys
    args = parse_args()
    sys.path.append(args.base_root)

    dataset_test = ShapeNetSegDataset(root=args.dir_dataset, split='test')
    get_softmax_probability_first(args, dataset_test)

    for i_start, i_end in [[args.d_cutoff_start, args.d_cutoff_end]]:
        options = []
        for r_cut_base in range(i_start, i_end):
            options += ['%s_%d' % (x, r_cut_base) for x in ['uniform', 'linear', 'cosine', 'gaussian', 'exponential']]
        print(options)
        with Pool(len(options)) as p:
            p.map(pcs_softmax, options)