import re
import io
import lmdb
from utils import *
from PIL import Image
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import gc

def build_img_dataloader(args, dataset):
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    loader_kwargs['collate_fn'] = collate_fn_nbs
    if args.num_workers > 1:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    loader_kwargs_val = deepcopy(loader_kwargs)
    loader_kwargs_test = deepcopy(loader_kwargs)

    if args.batch_size > 1:
        loader_kwargs['collate_fn'] = collate_fn_img_batch

    keep_same_psize = args.same_psize

    if args.datasets.lower().startswith('surv'):
        keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
        keep_same_psize = keep_same_psize - keep_same_psize_sub
        _train_set_1 = SurvLMDBDataset(args, args.dataset_sub_root, dataset['train'], persistence=args.persistence,
                                       keep_same_psize=keep_same_psize_sub, mode="train", img_size=args.img_size)
        _train_set_2 = SurvLMDBDataset(args, args.env_train, dataset['train'], persistence=args.persistence,
                                       keep_same_psize=keep_same_psize, mode="train", img_size=args.img_size)
        train_set = MultiScaleDataset(_train_set_1, _train_set_2)

        _test_img_size = 224
        test_set = SurvLMDBDataset(args, args.env, dataset['test'], persistence=args.persistence, mode="test",
                                   img_size=_test_img_size)
        if args.val_ratio != 0.:
            val_set = SurvLMDBDataset(args, args.env, dataset['val'], persistence=args.persistence, mode="val",
                                      img_size=_test_img_size)
        else:
            val_set = SurvLMDBDataset(args, args.env, dataset['test'], persistence=args.persistence, mode="val",
                                      img_size=_test_img_size)
    else:
        keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
        keep_same_psize = keep_same_psize - keep_same_psize_sub
        _train_set_1 = ClsLMDBDataset(args, args.dataset_sub_root, dataset['train'], persistence=args.persistence,
                                      keep_same_psize=keep_same_psize_sub, mode="train", _type=args.datasets,
                                      img_size=args.img_size)
        _train_set_2 = ClsLMDBDataset(args, args.env_train, dataset['train'], persistence=args.persistence,
                                      keep_same_psize=keep_same_psize, mode="train", _type=args.datasets,
                                      img_size=args.img_size)
        train_set = MultiScaleDataset(_train_set_1, _train_set_2)

        _test_img_size = 224
        test_set = ClsLMDBDataset(args, args.env, dataset['test'], persistence=args.persistence, _type=args.datasets,
                                  mode="test", img_size=_test_img_size)
        if args.val_ratio != 0.:
            val_set = ClsLMDBDataset(args, args.env, dataset['val'], persistence=args.persistence, _type=args.datasets,
                                     mode="val", img_size=_test_img_size)
        else:
            val_set = ClsLMDBDataset(args, args.env, dataset['test'], persistence=args.persistence, _type=args.datasets,
                                     mode="val", img_size=_test_img_size)

    if args.distributed:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=args.drop_last, **loader_kwargs,
                                  sampler=DistributedSampler(train_set, shuffle=True))
        val_loader = DataLoader(val_set, batch_size=1, **loader_kwargs_val,
                                sampler=SequentialDistributedSampler(val_set, batch_size=1))
        test_loader = DataLoader(test_set, batch_size=1, **loader_kwargs_test,
                                 sampler=SequentialDistributedSampler(test_set, batch_size=1))
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,
                                  **loader_kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_kwargs_val)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_kwargs_test)

    # Setting fp16 here will affect performance
    train_loader = PrefetchLoader(train_loader, device=args.device, need_transform=args.img_transform != 'none',
                                  transform_type=args.img_transform, img_size=args.img_size,
                                  trans_chunk=args.img_trans_chunk, crop_scale=args.crop_scale,
                                  load_gpu_later=args.load_gpu_later_train)
    val_loader = PrefetchLoader(val_loader, device=args.device, is_train=False, load_gpu_later=args.load_gpu_later,
                                trans_chunk=args.img_trans_chunk)
    test_loader = PrefetchLoader(test_loader, device=args.device, is_train=False, load_gpu_later=args.load_gpu_later,
                                 trans_chunk=args.img_trans_chunk)

    return train_loader, val_loader, test_loader

def update_worker_dict(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.update_imgs()


class ClsLMDBDataset(Dataset):
    def __init__(self, args, env, df, keep_same_psize=0, mode="train", seed=2021, transforms=None, _type='brca',
                 persistence=False, channels_last=True, img_size=224):
        super(ClsLMDBDataset, self).__init__()
        self.img_size = img_size
        self.rows = df
        self.lmdb_path = env
        self.env = self.open_lmdb()
        self.seed = seed
        self.keep_same_psize = keep_same_psize
        self.mode = mode
        self.transforms = transforms
        self.args = args
        self.current_epoch = 0

        with self.env.begin(write=False) as txn:
            # key1_sub1, key1_sub2, key2_sub1, key2_sub2,...
            self.PN_dict = pickle.loads(txn.get(b'__pn__'))
            self.all_slides = pickle.loads(txn.get(b'__slide__'))

        if hasattr(self, 'lmdb_path'):
             self.close_lmdb()
             self.env = None

        # Create a boolean mask to mark rows to keep
        new_rows = []
        for idx, row in self.rows.iterrows():
            current_id = row['ID']
            # Condition 1: Exact match
            if current_id in self.all_slides:
                new_rows.append(row)
                continue

            # Condition 2: Prefix match
            matched_slides = [slide for slide in self.all_slides if slide.startswith(current_id)]

            # If there are prefix matches
            if matched_slides:
                # Create a new row for each matched slide
                for matched_slide in matched_slides:
                    new_row = row.copy()  # Copy all data from original row
                    new_row['ID'] = matched_slide  # Update ID to matched slide
                    new_rows.append(new_row)

        # Convert all kept rows to new DataFrame
        self.rows = pd.DataFrame(new_rows)
        self.rows.reset_index(drop=True, inplace=True)

        # Get all related image files under case ID, store in dictionary
        self.all_imgs = {}
        # Always keep complete patch information without dropping patches
        self.all_imgs_all = {}
        self.all_pos = {}

        for i, row in self.rows.iterrows():
            _slide = row['ID']
            _slide_specific_keys = [str(_slide) + '-' + str(i) for i in range(self.PN_dict[_slide])]

            self.all_imgs_all[str(_slide)] = _slide_specific_keys
            self.all_imgs[str(_slide)] = _slide_specific_keys

        if mode == 'train':
            args.all_imgs_train = self.all_imgs
            args.all_imgs_all_train = self.all_imgs_all
        elif mode == 'val':
            args.all_imgs_val = self.all_imgs
        elif mode == 'test':
            args.all_imgs_test = self.all_imgs

        # label
        if _type.lower().startswith('bio'):
            self.rows.loc[:, 'Label'] = self.rows['Label'].astype(int)
        else:
            if 'brca' in _type.lower():
                self.rows.loc[:, 'Label'] = self.rows['Label'].map({'IDC': 0, 'ILC': 1})
            elif re.search(r'panda', _type.lower()) is not None:
                self.rows.loc[:, 'Label'] = self.rows['Label'].astype(int)
            elif re.search(r'call', _type.lower()) is not None:
                self.rows.loc[:, 'Label'] = self.rows['Label'].astype(int)
                pass
            else:
                raise NotImplementedError

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def open_lmdb(self):

        GB = 1024 * 1024 * 1024  # 1 GB in bytes

        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                             map_size=100 * GB)

        return self.env

    def update_imgs(self):
        if self.mode == 'train':
            self.all_imgs = self.args.all_imgs_train
        elif self.mode == 'val':
            self.all_imgs = self.args.all_imgs_val
        elif self.mode == 'test':
            self.all_imgs = self.args.all_imgs_test

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
        return None

    def load_data(self, slide_name, valid_imgs, length, keys):
        imgs = {}
        with self.env.begin(write=False, buffers=True) as txn:
            cursor = txn.cursor()
            cursor.set_key(u"{}-0".format(slide_name).encode('ascii'))
            for i, (_key, _value) in enumerate(cursor.iternext()):
                if i == length:
                    break
                img_data = pickle.loads(_value.tobytes())
                imgs[keys[i]] = Image.open(io.BytesIO(img_data))

        imgs = dict(sorted(imgs.items(), key=lambda x: x[0], reverse=False))

        return list(imgs.values())

    def load_data_random(self, valid_imgs):
        imgs = torch.empty((len(valid_imgs), 3, self.img_size, self.img_size), memory_format=torch.channels_last)
        with self.env.begin(write=False, buffers=True) as txn:
            for i, key_str in enumerate(valid_imgs):
                imgs[i] = torch.from_numpy(
                    imfrombytes(pickle.loads(txn.get(key_str.encode('ascii')).tobytes())).transpose(2, 0, 1))

        return imgs

    def __del__(self):
        if hasattr(self, 'lmdb_path'):
            self.close_lmdb()
            self.env = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        slide_id = row['ID']
        label = row['Label']
        need_close = False
        if self.env is None:
            self.env = self.open_lmdb()
            need_close = True  
    
        try:
            if self.keep_same_psize > 0.:
                _all_imgs = get_same_psize_img(self.args, self.all_imgs_all[slide_id], 
                                         self.keep_same_psize, self.all_imgs[slide_id])
            else:
                _all_imgs = self.all_imgs[slide_id]
        
            imgs = self.load_data_random(_all_imgs)

            outputs = {'input': imgs, 'target': torch.tensor(label).unsqueeze(0)}
        
            return outputs
    
        finally:
            if need_close and self.env is not None:
                self.env.close()
                self.env = None


class SurvLMDBDataset(Dataset):
    def __init__(self, args, env, df, keep_same_psize=0, mode="train", seed=2021, transforms=None, persistence=False,
                 channels_last=False, img_size=224):
        super(SurvLMDBDataset, self).__init__()

        self.rows = df
        self.img_size = img_size
        if isinstance(env, str):
            self.lmdb_path = env
            self.env = self.open_lmdb()
        else:
            self.env = env

        self.seed = seed
        self.keep_same_psize = keep_same_psize
        self.mode = mode
        self.transforms = transforms
        self.args = args
        self.current_epoch = 0

        with self.env.begin(write=False) as txn:
            self.PN_dict = pickle.loads(txn.get(b'__pn__'))
            self.all_slides = pickle.loads(txn.get(b'__slide__'))

        if hasattr(self, 'lmdb_path'):
            self.close_lmdb()
            self.env = None

        self.slide_name = {}
        self.all_imgs = {}
        self.all_imgs_all = {}
        self.all_pos = {}

        for index, row in self.rows.iterrows():
            case_name = row['ID']
            slides = [slide for slide in self.all_slides if case_name in slide]

            self.slide_name[str(case_name)] = slides

            _slide_specific_keys = []
            _seq_pos = []
            for _slide in slides:
                _slide_specific_keys += [str(_slide) + '-' + str(i) for i in range(self.PN_dict[_slide])]

            self.all_imgs_all[str(case_name)] = _slide_specific_keys
            self.all_imgs[str(case_name)] = _slide_specific_keys

        self.rows = self.rows[self.rows['ID'].apply(lambda x: x in self.slide_name and bool(self.slide_name[x]))]
        self.rows.reset_index(drop=True, inplace=True)

        if mode == 'train':
            args.all_imgs_train = self.all_imgs
            args.all_imgs_all_train = self.all_imgs_all
        elif mode == 'val':
            args.all_imgs_val = self.all_imgs
        elif mode == 'test':
            args.all_imgs_test = self.all_imgs

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def open_lmdb(self):
        GB = 1024 * 1024 * 1024
        return lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                         map_size=100 * GB)

    def update_imgs(self):
        if self.mode == 'train':
            self.all_imgs = self.args.all_imgs_train
        elif self.mode == 'val':
            self.all_imgs = self.args.all_imgs_val
        elif self.mode == 'test':
            self.all_imgs = self.args.all_imgs_test

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
        return None

    def load_data_random(self, valid_imgs):
        imgs = torch.empty((len(valid_imgs), 3, self.img_size, self.img_size), memory_format=torch.channels_last)
        with self.env.begin(write=False, buffers=True) as txn:
            try:
                for i, key_str in enumerate(valid_imgs):
                    imgs[i] = torch.from_numpy(
                        imfrombytes(pickle.loads(txn.get(key_str.encode('ascii')).tobytes())).transpose(2, 0, 1))
            except:
                print(key_str)
        return imgs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        Study, case_name, Event, Status, label = self.rows.loc[
            idx, ['Study', 'ID', 'Event', 'Status', 'Label']].values.tolist()
        Censorship = 1 if int(Status) == 0 else 0

        if self.env is None:
            self.env = self.open_lmdb()

        if self.keep_same_psize > 0.:
            _all_imgs = get_same_psize_img(self.args, self.all_imgs_all[case_name], self.keep_same_psize,
                                           self.all_imgs[case_name])

        else:
            _all_imgs = self.all_imgs[case_name]

        imgs = self.load_data_random(_all_imgs)

        outputs = {
            "input": imgs,
            "event": torch.tensor(Event).unsqueeze(0),
            "censorship": torch.tensor(Censorship).unsqueeze(0),
            "target": torch.tensor(label).unsqueeze(0)
        }

        return outputs

    def __del__(self):
        if hasattr(self, 'lmdb_path'):
            self.close_lmdb()


class MultiScaleDataset(Dataset):
    def __init__(self, dataset1, dataset2, image_input=True, shuffle=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.image_input = image_input
        self.shuffle = shuffle
        assert len(dataset1) == len(dataset2), "Datasets must have equal length"

    def __len__(self):
        return len(self.dataset1)

    def set_epoch(self, epoch):
        self.dataset1.set_epoch(epoch)
        self.dataset2.set_epoch(epoch)

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]

        output = {}

        input1, input2 = data1['input'], data2['input']

        if self.image_input:
            batch_size1, c1, h1, w1 = input1.shape
            batch_size2, c2, h2, w2 = input2.shape

            is_channels_last = input1.is_contiguous(memory_format=torch.channels_last)

            combined_input = torch.empty(
                (batch_size1 + batch_size2, c1, h1, w1),
                dtype=input1.dtype,
                device=input1.device,
                memory_format=torch.channels_last if is_channels_last else torch.contiguous_format
            )
            combined_input[:batch_size1] = input1
            combined_input[batch_size1:] = input2

        else:
            combined_input = torch.cat([input1, input2], dim=0)

        if self.shuffle:
            indices = torch.randperm(combined_input.size(0))
            combined_input = combined_input[indices]

        output['input'] = combined_input
        output['target'] = data1['target']

        if 'idx' in data1:
            output['idx'] = data1['idx']

        if 'censorship' in data1:
            output['censorship'] = data1['censorship']

        if 'event' in data1:
            output['event'] = data1['event']

        return output


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class PrefetchLoader:
    def __init__(
            self,
            loader,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            channels=3,
            device=torch.device('cuda'),
            img_dtype=torch.float32,
            need_norm=True,
            need_transform=False,
            transform_type='strong',
            img_size=224,
            is_train=True,
            trans_chunk=4,
            crop_scale=0.08,
            load_gpu_later=False):

        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.need_norm = need_norm

        if need_transform:
            if is_train:
                if transform_type == 'strong':
                    ta_list = [
                        ta_transforms.RandomHorizontalFlip(p=0.5),
                        ta_transforms.RandomVerticalFlip(p=0.5),
                    ]
                elif transform_type == 'strong_v2':
                    ta_list = [
                        ta_transforms.RandomHorizontalFlip(p=0.5),
                        ta_transforms.RandomVerticalFlip(p=0.5),
                        ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
                    ]
                elif transform_type == 'weak_strong':
                    ta_list = [
                        ta_transforms.RandomAffine(degrees=0, translate=(100 / 256, 100 / 256)),
                    ]
                else:
                    ta_list = [
                    ]
                if img_size != 224:
                    ta_list += [
                        ta_transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                    ]
                else:
                    ta_list += [
                        ta_transforms.Resize(224),
                    ]
                self.transform = ta_transforms.SequentialTransform(
                    ta_list,
                    inplace=True,
                    batch_inplace=True,
                    batch_transform=True,
                    num_chunks=trans_chunk,
                    permute_chunks=False,
                )
            else:
                self.transform = ta_transforms.SequentialTransform(
                    [
                        ta_transforms.CenterCrop(224),
                    ],
                    inplace=True,
                    batch_inplace=True,
                    batch_transform=True,
                    num_chunks=1,
                    permute_chunks=False,
                )
        else:
            self.transform = None
        self.device = self.device_input = device
        self.img_dtype = img_dtype
        self.no_gpu_key = []
        if load_gpu_later:
            device = torch.device('cpu')
            self.device_input = device
            self.no_gpu_key = ['input']

        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_batch in self.loader:
            with stream_context():
                next_batch = {
                    k: ([v.to(device=self.device_input, non_blocking=True) for v in v] if isinstance(v,
                                                                                                     list) and k == 'input' else
                        v.to(device=self.device, non_blocking=True) if isinstance(v,
                                                                                  torch.Tensor) and k not in self.no_gpu_key else v)
                    for k, v in next_batch.items()
                }
                if self.transform is not None:
                    if isinstance(next_batch['input'], list):
                        next_batch['input'] = [self.transform(tensor) for tensor in next_batch['input']]
                    else:
                        next_batch['input'] = self.transform(next_batch['input'])
                if self.need_norm:
                    if isinstance(next_batch['input'], list):
                        next_batch['input'] = [tensor.to(self.img_dtype).sub_(self.mean).div_(self.std) for tensor in
                                               next_batch['input']]
                    else:
                        next_batch['input'] = next_batch['input'].to(self.img_dtype).sub_(self.mean).div_(self.std)

            if not first:
                yield batch
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset