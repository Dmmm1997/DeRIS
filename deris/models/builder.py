from mmcv.utils import Registry


VIS_ENCODERS = Registry("VIS_ENCS")
MODELS = Registry("MODELS")
HEADS = Registry("HEADS")
BRANCHS = Registry("BRANCHS")
LAN_ENCODERS = Registry("LAN_ENCS")


def build_vis_enc(cfg):
    """Build vis_enc."""
    return VIS_ENCODERS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_model(cfg):
    """Build model."""
    return MODELS.build(cfg)


def build_branch(cfg):
    """Build model."""
    return BRANCHS.build(cfg)


def build_lan_enc(cfg):
    """Build lan_enc."""
    return LAN_ENCODERS.build(cfg)
