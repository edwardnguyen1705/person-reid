from .adb_attension import CAM, PAM
from .cbam import CbamModule
from .fpb_attention import FPBAttention
from .global_context import GlobalContext
from .non_local import NonLocalBlock
from .plr_attention import PLRAttention
from .semodule import SEModule


def build_attention(
    name, channels: int, cam_batchnorm: bool, pam_batchnorm: bool, rd_ratio: float
):
    if name == "CAM":
        return CAM(channels=channels, batchnorm=cam_batchnorm)
    elif name == "PAM":
        return PAM(channels=channels, batchnorm=pam_batchnorm)
    elif name == "CBAM":
        return CbamModule(channels=channels)
    elif name == "FPBAttention":
        return FPBAttention(
            channels=channels, cam_batchnorm=cam_batchnorm, pam_batchnorm=pam_batchnorm
        )
    elif name == "GlobalContext":
        return GlobalContext(channels=channels, rd_ratio=rd_ratio)
    elif name == "NonLocalBlock":
        return NonLocalBlock(channels=channels, rd_ratio=rd_ratio)
    elif name == "PLRAttention":
        return PLRAttention(channels=channels, rd_ratio=rd_ratio)
    elif name == "SEModule":
        return SEModule(channels=channels, rd_ratio=rd_ratio)
    else:
        raise ValueError("name not support, got unexpected: {}".format(name))
