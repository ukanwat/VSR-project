import time
import os.path as osp
import megengine as mge
from megengine.module import Module
from .base_runner import BaseRunner, module_ckpt_suffix, optim_ckpt_suffix
from edit.utils import is_list_of, mkdir_or_exist


class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader):
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(0.05)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.losses = self.model.train_step(
                data_batch, self._epoch, self._iter)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def test(self, data_loader):
        self.mode = 'test'
        self.data_loader = data_loader
        self.call_hook('before_test_epoch')
        time.sleep(0.05)
        save_path = osp.join(self.work_dir, "test_results")
        mkdir_or_exist(save_path)
        for i, data_batch in enumerate(data_loader):
            batchdata = data_batch
            self._inner_iter = i
            self.call_hook('before_test_iter')
            self.outputs = self.model.test_step(batchdata,
                                                save_image=True,
                                                save_path=save_path)
            self.call_hook('after_test_iter')
            self._iter += 1

        self.call_hook('after_test_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training and test.
            workflow : train or test
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert workflow in ('test', 'train')
        assert len(
            data_loaders) == 1, "only support just length one data_loaders now"

        self._max_epochs = max_epochs
        if workflow == 'train':
            self._max_iters = self._max_epochs * len(data_loaders[0])
            self._iter = self.epoch * len(data_loaders[0])
            self.logger.info("{} iters for one epoch, trained iters: {}, total iters: {}".format(
                len(data_loaders[0]), self._iter, self._max_iters))

        else:
            assert max_epochs in (1, 4, 8)

        self.logger.info("Start running, work_dir: {}, workflow: {}, max epochs : {}".format(
            self.work_dir, workflow, max_epochs))
        self.logger.info("registered hooks: " + str(self.hooks))

        self.call_hook('before_run')
        while self.epoch < max_epochs:
            if isinstance(workflow, str):
                if not hasattr(self, workflow):
                    raise ValueError(
                        f'runner has no method named "{workflow}" to run an epoch')
                epoch_runner = getattr(self, workflow)
            else:
                raise TypeError(
                    'mode in workflow must be a str, but got {}'.format(type(workflow)))
            epoch_runner(data_loaders[0])

        time.sleep(0.05)

        self.call_hook('after_run')

    def resume(self, checkpoint, resume_optimizer=True):
        assert 'epoch_' in checkpoint
        res_dict = self.load_checkpoint(
            checkpoint, load_optim=resume_optimizer)

        self._epoch = res_dict['nums']
        self.logger.info("resumed from epoch: {}".format(self._epoch))

        if resume_optimizer:
            self.logger.info("load optimizer's state dict")
            for submodule_name in self.optimizers_cfg.keys():
                self.model.optimizers[submodule_name].load_state_dict(
                    res_dict[submodule_name])

    def save_checkpoint(self, out_dir, create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        filename_tmpl = "epoch_{}"
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        self.logger.info('save checkpoint to {}'.format(filepath))
        mkdir_or_exist(filepath)
        if isinstance(self.model.optimizers, dict):
            for key in self.model.optimizers.keys():
                submodule = getattr(self.model, key, None)
                assert submodule is not None, "model should have submodule {}".format(
                    key)
                assert isinstance(
                    submodule, Module), "submodule should be instance of megengine.module.Module"
                mge.save(submodule.state_dict(), osp.join(
                    filepath, key + module_ckpt_suffix))
                mge.save(self.model.optimizers[key].state_dict(
                ), osp.join(filepath, key + optim_ckpt_suffix))
        else:
            raise TypeError(
                " the type of optimizers should be dict for save_checkpoint")

        if create_symlink:
            pass

    def register_training_hooks(self,
                                lr_config,
                                checkpoint_config,
                                log_config):
        """Register default hooks for epoch-based training.

        Default hooks include:

        - LrUpdaterHook
        - CheckpointSaverHook
        - logHook
        """
        if lr_config is not None:
            lr_config.setdefault('by_epoch', True)
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', True)
        if log_config is not None:
            log_config.setdefault('by_epoch', False)

        self.register_checkpoint_hook(checkpoint_config)
        self.register_logger_hooks(log_config)
