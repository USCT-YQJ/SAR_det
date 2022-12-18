import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import cv2
import torch
import random
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
from .rotate_aug import RotateAugmentation
from .rotate_aug import RotateTestAugmentation


class Custom2Dataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 rotate_aug=None,
                 rotate_test_aug=None,
                 resize_keep_ratio=True,
                 test_keep_size=False,
                 test_mode=False,
                 check_mask=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_size = img_scale[0]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode
        # check mask error
        self.check_mask = check_mask
        self.test_keep_size = test_keep_size

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # if use rotation augmentation
        if rotate_aug is not None:
            self.rotate_aug = RotateAugmentation(self.CLASSES, **rotate_aug)
        else:
            self.rotate_aug = None

        if rotate_test_aug is not None:
            #  dot not support argument settings currently
            self.rotate_test_aug = RotateTestAugmentation()
        else:
            self.rotate_test_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_cat_ids(self, idx):
        """Get category ids by index.
        Args:
            idx (int): Index of data.
        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.get_ann_info(idx)['labels'].astype(np.int).tolist()

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = x[0] + padw  # top left x
        y[1] = x[1] + padh  # top left y
        y[2] = x[2] + padw  # bottom right x
        y[3] = x[3] + padh  # bottom right y
        for i in range(4):
            y[i]=(int)(y[i]/2)
            if y[i]<0:
                y[i]=0
            if y[i]>w:  # w == h
                y[i]=w

        # y[0] = (int)(((x0 + x1)/2)/2)  # center x
        # y[1] = (int)(((y0 + y1)/2)/2) # center y
        # y[2] = (x1 - x0)/2  # w
        # y[3] = (y1 - y0)/2  # h
    
        return y
    def draw(self, img, bboxes):
        print("==============bbox=============")
        print(bboxes)
        for bbox in bboxes:
            print(bbox)
            lu = (int) (bbox[0])
            # lux = (int)(bbox[0]-bbox[2]/2)
            # luy = (int)(bbox[1]-bbox[3]/2)
            # brx = (int)(bbox[0]+bbox[2]/2)
            # bry = (int)(bbox[1]+bbox[3]/2)
            img = cv2.rectangle(img, ((int)(bbox[0]),(int)(bbox[1])), ((int)(bbox[2]),(int)(bbox[3])), (255, 0, 255), 2)
        cv2.imwrite("test"+str(random.randint(0,5))+".jpg", img)
    
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        if self.with_mask:
            gt_masks = ann['masks']
        # print("===================annos==============")
        # print("===================bboxes==============")
        # print(gt_bboxes)
        # print("===================labels==============")
        # print(gt_labels)
        # print("===================mask================")
        # print(gt_masks)
        self.mosaic = True
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        do_mos = random.random()
        if self.mosaic and do_mos<=0.4:
            bboxes4 = []
            label4 = []
            mask4 = []
            s = self.img_size
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
            sample_list = [i for i in range(len(self.img_infos))]
            indices = [idx] + random.choices(sample_list, k=3)  # 3 additional image indices
            random.shuffle(indices)
            for i, index in enumerate(indices):
            # Load image
                ann1 = self.get_ann_info(index)
                ann1_bbox = np.copy(ann1['bboxes'])
                ann1_label = np.copy(ann1['labels'])
                ann1_mask = ann1['masks']
                img_info = self.img_infos[index]
                img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
                h, w = img.shape[:2]
                # img, _, (h, w) = self.load_image(index)
                # place img in img4
                # print("=============s===========")
                # print(s)
                # print(img.shape[2])
                if i == 0:  # top left
                    img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                for i in range(len(ann1_mask)):
                    mask4_no = np.full((s * 2, s * 2), 114, dtype=np.uint8)
                    mask4_no[y1a:y2a, x1a:x2a] = ann1_mask[i][y1b:y2b, x1b:x2b]
                    mask4_no = cv2.resize(mask4_no, (s, s))
                    mask4.append(mask4_no)
                padw = x1a - x1b
                padh = y1a - y1b
                for bbox in ann1_bbox:
                    temp = self.xywhn2xyxy(bbox, w, h, padw, padh)
                    bboxes4.append(temp)
                label4.append(ann1_label)
            # print("===================img size============")
            # print(img.shape[:2])
            # print("======================================")
            img4 = cv2.resize(img4, (s, s))
            # mask4 = cv2.resize(mask4, (s, s))
            label4 = np.concatenate(label4, 0)
            
            img = img4
            gt_bboxes = bboxes4
            # self.draw(img, gt_bboxes)
            gt_labels = label4
            if self.with_mask:
                gt_masks = mask4

        #
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # rotate augmentation
        if self.rotate_aug is not None:
            # only support mask now, TODO: support none mask version
            img, gt_bboxes, gt_masks, gt_labels = self.rotate_aug(img, gt_bboxes,
                                                                  gt_masks, gt_labels, img_info['filename'])

            gt_bboxes = np.array(gt_bboxes).astype(np.float32)
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
                return None
        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            # gt_masks = self.mask_transform(ann['masks'], pad_shape,
            #                                scale_factor, flip)
            gt_masks = self.mask_transform(gt_masks, pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        if self.check_mask:
            # filter annotations too small
            del_rows = []
            for i in range(gt_masks.shape[0]):
                contours, hierarchy = cv2.findContours(gt_masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) == 0:
                    del contours
                    del_rows.append(i)
            if len(del_rows) > 0:
                gt_bboxes = np.delete(gt_bboxes, del_rows, axis=0)
                gt_labels = np.delete(gt_labels, del_rows, axis=0)
                gt_masks = np.delete(gt_masks, del_rows, axis=0)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        # TODO: make the flip and rotate at the same time
        # TODO: when implement the img rotation, we do not consider the proposals, add it in future
        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                angle=0)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        def prepare_rotation_single(img, scale, flip, angle):
            _img, img_shape, pad_shape, scale_factor = self.rotate_test_aug(
                img, angle=angle)
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                _img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            # if self.rotate_test_aug is not None:
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                angle=angle
            )
            return _img, _img_meta

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            # to keep original size of image
            if self.test_keep_size:
                w,h = img.shape[:2]
                M = max(w,h)
                m = min(w,h)
                scale = (M,m)
            try:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, proposal)
            except AttributeError:
                img_path = osp.join(self.img_prefix, img_info['filename'])
                with open(img_info['filename'] + '.txt', 'w') as f:
                    f.write(img_path)

            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        if self.rotate_test_aug is not None:
            # rotation augmentation
            # do not support proposals currently
            # img_show = img.copy()
            # mmcv.imshow(img_show, win_name='original')
            for angle in [90, 180, 270]:

                for scale in self.img_scales:
                    _img, _img_meta, = prepare_rotation_single(
                        img, scale, False, angle)
                    imgs.append(_img)
                    img_metas.append(DC(_img_meta, cpu_only=True))
                    # proposals.append(_proposal)
                    if self.flip_ratio > 0:
                        _img, _img_meta = prepare_rotation_single(
                            img, scale, True, proposal, angle)
                        imgs.append(_img)
                        img_metas.append(DC(_img_meta, cpu_only=True))
                    # # # # TODO: rm if after debug
                    # if angle == 180:
                    #     img_show = _img.cpu().numpy().copy()
                    #     mmcv.imshow(img_show, win_name=str(angle))
                    # import pdb;pdb.set_trace()

        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
