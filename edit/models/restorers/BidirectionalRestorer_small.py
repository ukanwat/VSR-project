import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from tqdm import tqdm
from megengine.jit import trace


def get_bilinear(image):
    B, T, C, h, w = image.shape
    image = image.reshape(-1, C, h, w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B, T, C, 4*h, 4*w)


def train_generator_batch(image, label, *, gm, netG, netloss):
    B, T, _, h, w = image.shape
    biup = get_bilinear(image)

    HR_mask = 1
    netG.train()
    with gm:
        forward_hiddens = []
        backward_hiddens = []
        res = []

        image = image.reshape(B*T, 3, h, w)
        image = netG.rgb(image).reshape(B, T, -1, h, w)

        now_frame = image[:, 0, ...]
        hidden = now_frame
        forward_hiddens.append(now_frame)
        for i in range(1, T):
            now_frame = image[:, i, ...]
            hidden = netG.aggr(F.concat([hidden, now_frame], axis=1))
            forward_hiddens.append(hidden)

        now_frame = image[:, T-1, ...]
        hidden = now_frame
        backward_hiddens.append(now_frame)
        for i in range(T-2, -1, -1):
            now_frame = image[:, i, ...]
            hidden = netG.aggr(F.concat([hidden, now_frame], axis=1))
            backward_hiddens.append(hidden)

        for i in range(T):
            res.append(netG.upsample(
                F.concat([forward_hiddens[i], backward_hiddens[T-i-1]], axis=1)))

        res = F.stack(res, axis=1)
        res = res+biup
        loss = netloss(res, label, HR_mask)

        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss


def test_generator_batch(image, *, netG):
    B, T, _, h, w = image.shape
    biup = get_bilinear(image)
    netG.eval()
    forward_hiddens = []
    backward_hiddens = []
    res = []

    image = image.reshape(B*T, 3, h, w)
    image = netG.rgb(image).reshape(B, T, -1, h, w)

    now_frame = image[:, 0, ...]
    hidden = now_frame
    forward_hiddens.append(now_frame)
    for i in tqdm(range(1, T)):
        now_frame = image[:, i, ...]
        hidden = netG.aggr(F.concat([hidden, now_frame], axis=1))
        forward_hiddens.append(hidden)

    now_frame = image[:, T-1, ...]
    hidden = now_frame
    backward_hiddens.append(now_frame)
    for i in tqdm(range(T-2, -1, -1)):
        now_frame = image[:, i, ...]
        hidden = netG.aggr(F.concat([hidden, now_frame], axis=1))
        backward_hiddens.append(hidden)

    for i in tqdm(range(T)):
        res.append(netG.upsample(
            F.concat([forward_hiddens[i], backward_hiddens[T-i-1]], axis=1)))

    res = F.stack(res, axis=1)
    res = res+biup
    return res


epoch_dict = {}


def adjust_learning_rate(optimizer, epoch):
    if epoch >= 6 and epoch % 2 == 0 and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.7
        print("adjust lr! , now lr: {}".format(param_group["lr"]))


@MODELS.register_module()
class BidirectionalRestorer_small(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(BidirectionalRestorer_small, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        self.generator = build_backbone(generator)

        self.pixel_loss = build_loss(pixel_loss)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch, now_iter):
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        HR_tensor = mge.tensor(batchdata['gt'], dtype="float32")
        loss = train_generator_batch(
            LR_tensor, HR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        shift = self.eval_cfg.get('save_shift', 0)
        assert isinstance(key, str)
        L = key.split("/")
        return int(L[-1][:-4]), str(int(L[-2]) - shift).zfill(3)

    def test_step(self, batchdata, **kwargs):
        """
            possible kwargs:
                save_image
                save_path
                ensemble
        """
        lq = batchdata['lq']
        gt = batchdata.get('gt', None)
        assert len(batchdata['lq_path']) == 1
        lq_paths = batchdata['lq_path'][0]
        now_start_id, clip = self.get_img_id(lq_paths[0])
        now_end_id, _ = self.get_img_id(lq_paths[-1])
        assert clip == _
        if now_start_id == 0:
            print("first frame: {}".format(lq_paths[0]))
            self.LR_list = []
            self.HR_list = []

        B, _, origin_H, origin_W = lq.shape
        lq = img_multi_padding(
            lq, padding_multi=self.eval_cfg.multi_pad, pad_method="edge")
        self.LR_list.append(mge.tensor(lq, dtype="float32"))

        if gt is not None:
            for i in range(B):
                self.HR_list.append(gt[i:i+1, ...])

        if now_end_id == 99:
            print("start to forward all frames....")
            if self.eval_cfg.gap == 1:
                self.LR_list = F.concat(self.LR_list, axis=0)
                self.HR_G = test_generator_batch(F.expand_dims(
                    self.LR_list, axis=0), netG=self.generator)
            elif self.eval_cfg.gap == 2:
                raise NotImplementedError("not implement gap != 1 now")

            else:
                raise NotImplementedError("do not support eval&test gap value")

            scale = self.generator.upscale_factor

            self.HR_G = img_de_multi_padding(
                self.HR_G.numpy(), origin_H=origin_H * scale, origin_W=origin_W * scale)

            if kwargs.get('save_image', False):
                print("saving images to disk ...")
                save_path = kwargs.get('save_path')
                B, T, _, _, _ = self.HR_G.shape
                assert B == 1
                assert T == 100
                for i in tqdm(range(T)):
                    img = tensor2img(self.HR_G[0, i, ...], min_max=(0, 1))
                    if (i+1) % 10 == 0:
                        imwrite(img, file_path=os.path.join(
                            save_path, "partframes", f"{clip}_{str(i).zfill(8)}.png"))
                    imwrite(img, file_path=os.path.join(
                        save_path, "allframes", f"{clip}_{str(i).zfill(8)}.png"))

        return now_end_id == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        if gathered_outputs:
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_list) == 100
            res = []
            for i in range(len(self.HR_list)):
                G = tensor2img(self.HR_G[0, i, ...], min_max=(0, 1))
                gt = tensor2img(self.HR_list[i][0], min_max=(0, 1))
                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric +
                                "_RGB"] = self.allowed_metrics[metric](G, gt, crop_border)

                res.append(eval_result)
            return res
        else:
            return []
