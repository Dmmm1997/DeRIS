_base_ = [
    "../../_base_/datasets/segmentation/grefs.py",
    "../../_base_/misc.py",
]
dataset = "GRefCOCO"
max_token = 50
img_size = 224
hier_img_size = 384
patch_size = 16
num_query = 20

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFileGRES_TO",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
        turn_hard_negative={
            "enable": True,
            "rate": 0.15,
            "sim": True,
            "min_text_length": 0,
            "simlarity_threshold": 0.6,
        },
    ),
    # dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(
        type="HierResize",
        img_scale=(hier_img_size, hier_img_size),
        mini_img_scale=(img_size, img_size),
        keep_ratio=False,
    ),
    # dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=[
            "img",
            "img_mini",
            "ref_expr_inds",
            "text_attention_mask",
            "gt_mask_rle",
            "gt_bbox",
            "gt_mask_parts_rle",
        ],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
            "refer_target_index",
            "tokenized_words",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFileGRES_TO",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
    ),
    dict(
        type="HierResize",
        img_scale=(hier_img_size, hier_img_size),
        mini_img_scale=(img_size, img_size),
        keep_ratio=False,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "img_mini", "ref_expr_inds", "text_attention_mask", "gt_mask_rle", "gt_bbox", "gt_mask_parts_rle"],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
            "refer_target_index",
            "tokenized_words",
        ],
    ),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=val_pipeline,
    ),
    testA=dict(
        pipeline=test_pipeline,
    ),
    testB=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="MIXGrefUniModel_HierVG_MRLoopback",
    understanding_branch=dict(
        type="UnderstandingBranchLoopBack",
        vis_enc=dict(
            type="BEIT3",
            img_size=img_size,
            patch_size=patch_size,
            vit_type="base",
            drop_path_rate=0.1,
            vocab_size=64010,
            freeze_layer=-1,
            vision_embed_proj_interpolate=False,
            pretrain="pretrain_weights/beit3_base_patch16_224.zip",
        ),
        vis_enc_outdim=768,
        hidden_channels=256,
        freeze_vis_enc=False,  # 冻住 beit3
        num_classes=2,  # 2 这里改的话 head["class_weights"]
    ),
    perception_branch=dict(
        type="PerceptionBranchLoopBack",
        mask_config="deris/models/branchs/perception_branch/mask_config/maskformer2_swin_small_bs16_50ep.yaml",
        hidden_size=256,
        num_queries=num_query,
        pretrain_weights="pretrain_weights/model_final_1e7f22.pkl",
        freeze_hier_backbone=True,  # 冻住 hier backbone
        dec_layers=3,  # mask2former 是 9
        prediction_load_pretrain=True,  # prediction 加载预训练
        pixel_decoder_load_pretrain=True,  # FPN 加载预训练
    ),
    head=dict(
        type="SimpleHead_Independent_match",
        mask_config="deris/models/branchs/perception_branch/mask_config/maskformer2_swin_small_bs16_50ep.yaml",
        matching_cost_weight={"mask": 1.0, "bbox": 0.0, "cls": 1.0},
        loss_weight={"mask": 1.0, "bbox": 0.0, "existence": 1.0, "cls": 1.0, "aux": 0.2, "global_mask": 1.0},
        additional_detection_supervision={"enable": False, "loss_weight": 1.0, "box": 0.0, "mask": 1.0, "cls": 1.0},
        num_queries=num_query,
        stage_weights=[1.0, 1.0],
    ),
    post_params={
        "score_weighted": False,
        "mask_threshold": 0.5,
        "score_threshold": 0.7,
        "with_nms": False,
        "with_mask": True,
        "outmask_type": "instance",
    },
    visualize_params={"enable": False, "row_columns": (4, 5), "train_interval": 50, "val_interval": 5},
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
# work_dir = "work_dir/paper_exp/decoder_ablation/ViTBaseP32-1.0decoder-40ep-512hw-refcocounc"

lr = 0.0001
optimizer_config = dict(
    type="Adam",
    lr=lr,
    lr_vis_enc=lr / 2.0,
    lr_lan_enc=lr / 10.0,
    lr_hier_vis_enc=lr / 10.0,
    lr_pixel_decoder=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=0,
    decay_steps=[6],
    decay_ratio=0.1,
    max_epoch=6,
)

log_interval = 50

start_save_checkpoint = 5
