from .jepa3d_wrapper import Encoder as JEPA3D

ENCODERS = {
    'jepa3d': JEPA3D,
}

def build_encoder(name, **kwargs):
    return ENCODERS[name](**kwargs)