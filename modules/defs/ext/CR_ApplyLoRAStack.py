import os
from ..meta import MetaField
from ..formatters import calc_lora_hash

def _unwrap_input_value(value):
    """
    Safely unwraps a value that might be a single-element list,
    which is common for ComfyUI input data.
    """
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value

def get_cr_lora_info_from_widgets(input_data):
    """
    Parses the input_data from a 'CR LoRA Stack' node to find all LoRA info.
    Returns three lists: names, model_strengths, clip_strengths.
    """
    names, model_strengths, clip_strengths = [], [], []
    i = 1
    while True:
        # Get the raw values for the current LoRA
        lora_name_raw = input_data[0].get(f"lora_name_{i}")
        
        # Unwrap the value to get the actual string
        lora_name = _unwrap_input_value(lora_name_raw)

        if not lora_name or lora_name == "None":
            break

        # Get and unwrap the strength values
        model_strength = _unwrap_input_value(input_data[0].get(f"lora_wt_{i}", 1.0))
        clip_strength = _unwrap_input_value(input_data[0].get(f"clip_wt_{i}", 1.0))

        names.append(lora_name)
        model_strengths.append(model_strength)
        clip_strengths.append(clip_strength)
        
        i += 1
        
    return names, model_strengths, clip_strengths

def get_cr_lora_names_from_node(node_id, obj, prompt, extra_data, outputs, input_data):
    names, _, _ = get_cr_lora_info_from_widgets(input_data)
    return names if names else None

def get_cr_lora_hashes_from_node(node_id, obj, prompt, extra_data, outputs, input_data):
    names, _, _ = get_cr_lora_info_from_widgets(input_data)
    return [calc_lora_hash(name) for name in names] if names else None

def get_cr_lora_strength_model_from_node(node_id, obj, prompt, extra_data, outputs, input_data):
    _, model_strengths, _ = get_cr_lora_info_from_widgets(input_data)
    return model_strengths if model_strengths else None

def get_cr_lora_strength_clip_from_node(node_id, obj, prompt, extra_data, outputs, input_data):
    _, _, clip_strengths = get_cr_lora_info_from_widgets(input_data)
    return clip_strengths if clip_strengths else None

CAPTURE_FIELD_LIST = {
    "CR LoRA Stack": {
        MetaField.LORA_MODEL_NAME:     {"selector": get_cr_lora_names_from_node},
        MetaField.LORA_MODEL_HASH:     {"selector": get_cr_lora_hashes_from_node},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_cr_lora_strength_model_from_node},
        MetaField.LORA_STRENGTH_CLIP:  {"selector": get_cr_lora_strength_clip_from_node},
    },
}
