from .target_attack import TargetAttack
from .embedd_attack import EmbeddAttack
from .one_pixel import OnePixelAttack
from .random import GaussNoiseAttack
from .color_attack import ColorAttack


def get_attack_from_config(model, attack_config):
    
    at_class = attack_config.attack
    attack_class = globals().get(at_class)

    return attack_class(model, **attack_config.params)