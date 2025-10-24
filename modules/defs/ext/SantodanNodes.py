from ..meta import MetaField
from ..formatters import calc_model_hash, calc_vae_hash

try:
    from ..formatters import calc_clip_hash
except ImportError:
    def calc_clip_hash(name):
        return f"hash_for_{name}"

def get_model_name(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the name of the primary model file being used."""
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    key = "ckpt_name" if mode == "full_checkpoint" else "base_model"
    return input_data[0].get(key, [None])[0]

def get_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the hash of the primary model file."""
    model_name = get_model_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if model_name:
        return calc_model_hash(model_name)
    return None

def get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the separate VAE name, ONLY in separate_components mode."""
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("vae_model", [None])[0]
    return None

def get_vae_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the separate VAE hash, ONLY in separate_components mode."""
    vae_name = get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if vae_name:
        return calc_vae_hash(vae_name)
    return None

def get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns a list of CLIP names, ONLY in separate_components mode."""
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        clip_names = []
        for key in ["clip_model_1", "clip_model_2", "clip_model_3"]:
            name = input_data[0].get(key, [None])[0]
            if name and name != "None":
                clip_names.append(name)
        return clip_names if clip_names else None
    return None

def get_clip_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns a list of CLIP hashes, ONLY in separate_components mode."""
    names = get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data)
    if names:
        return [calc_clip_hash(name) for name in names]
    return None

def get_clip_type(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the CLIP type, ONLY in separate_components mode."""
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("clip_type", [None])[0]
    return None

def get_unet_dtype(node_id, obj, prompt, extra_data, outputs, input_data):
    """Returns the UNet weight dtype, ONLY in separate_components mode."""
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("weight_dtype", [None])[0]
    return None

CAPTURE_FIELD_LIST = {
    "ModelAssembler": {
        MetaField.MODEL_NAME: {"selector": get_model_name},
        MetaField.MODEL_HASH: {"selector": get_model_hash},
        MetaField.VAE_NAME: {"selector": get_vae_name},
        MetaField.VAE_HASH: {"selector": get_vae_hash},
        "Clip Model Name(s)": {"selector": get_clip_names},
        "Clip Model Hash(es)": {"selector": get_clip_hashes},
        "Clip Type": {"selector": get_clip_type},
        "UNet Weight Type": {"selector": get_unet_dtype},
    }
}
