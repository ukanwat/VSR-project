dataroot = "/kaggle/input/redsds/test/test_sharp_bicubic"
load_path = './ckpt/epoch_62'
exp_name = 'mai_test_for_validation'
eval_part = tuple(map(str, range(240, 270)))


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

eval_cfg = dict(metrics=['PSNR'], crop_border=0,
                multi_pad=1, gap=1, save_shift=240)
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])

test_dataset_type = 'SRManyToManyDataset'
test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection",
         many2many=False, index_start=0, name_padding=True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq', 'num_input_frames', 'LRkey', 'lq_path'])
]


repeat_times = 1
data = dict(
    test_samples_per_gpu=5,
    test_workers_per_gpu=5,
    test=dict(
        type=test_dataset_type,
        lq_folder=dataroot + "/X4",
        num_input_frames=1,
        pipeline=test_pipeline,
        scale=scale,
        mode="test",
        eval_part=eval_part)
)

optimizers = dict(generator=dict(type='Adam', lr=0.00001))
total_epochs = 400 // repeat_times


lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', average_length=100),

    ])
evaluation = dict(interval=1, save_image=False,
                  multi_process=False, ensemble=False)


work_dir = f'./workdirs/{exp_name}'
load_from = load_path
resume_from = None
resume_optim = True
workflow = 'test'


log_level = 'INFO'
