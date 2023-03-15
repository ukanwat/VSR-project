exp_name = 'mai_training'
dataroot = "/mnt/tmp/REDS/train"
load_path = None
samples_per_gpu = 8
workers_per_gpu = 4
init_lr = 2 * 1e-3


scale = 4


model = dict(
    type='BidirectionalRestorer_small',
    generator=dict(
        type='BasicVSR_v5',
        in_channels=3,
        out_channels=3,
        hidden_channels=8,
        upscale_factor=scale),
    pixel_loss=dict(type='L2Loss'))


train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad=1, gap=1)
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])


train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[
         1], many2many=True, index_start=0, name_padding=True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        make_bin=False),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged',
        make_bin=False),
    dict(type='PairedRandomCrop', gt_patch_size=[90 * 4, 160 * 4]),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'],
         flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'lq_path', 'gt_path'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection",
         many2many=False, index_start=0, name_padding=True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt',
         'num_input_frames', 'LRkey', 'lq_path'])
]

repeat_times = 1
eval_part = tuple(map(str, range(240, 270)))
data = dict(

    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=dataroot + "/train_sharp_bicubic/X4",
            gt_folder=dataroot + "/train_sharp",
            num_input_frames=21,
            pipeline=train_pipeline,
            scale=scale,
            eval_part=eval_part)),

    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        lq_folder=dataroot + "/train_sharp_bicubic/X4",
        gt_folder=dataroot + "/train_sharp",
        num_input_frames=1,
        pipeline=eval_pipeline,
        scale=scale,
        mode="eval",
        eval_part=eval_part)
)


optimizers = dict(generator=dict(type='Adam', lr=init_lr, betas=(0.9, 0.999)))


total_epochs = 400 // repeat_times


lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', average_length=100),

    ])
evaluation = dict(interval=750, save_image=False,
                  multi_process=False, ensemble=False)


work_dir = f'./workdirs/{exp_name}'
load_from = load_path
resume_from = None
resume_optim = True
workflow = 'train'


log_level = 'INFO'
