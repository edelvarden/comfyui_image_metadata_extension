#https://github.com/Light-x02/ComfyUI-FluxSettingsNode
import json
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip

# ref format:
# '[{"name":"cecilia_shiro seijo to kuro bokushi_IllustriousXL_last","weight":0.5,"text_encoder_weight":0.5,
# "lora":"cecilia_shiro seijo to kuro bokushi_IllustriousXL_last.safetensors","loraWorks":""}]'
def get_lora_model_name_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["name"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_strength_model_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["weight"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_strength_clip_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([item["text_encoder_weight"] for item in lora_data])
        return lora_names
    else:
        return []

def get_lora_model_hash_stack(node_id, obj, prompt, extra_data, outputs, input_data):
    toggled_on = input_data[0]["lora_str"]

    if toggled_on:
        lora_names = []
        for lora_str in input_data[0]["lora_str"]:
            if lora_str == "":
                continue
            lora_data = json.loads(lora_str)
            lora_names.extend([calc_lora_hash(item["lora"], input_data) for item in lora_data])
        return lora_names
    else:
        return []

CAPTURE_FIELD_LIST = {
    "WeiLinComfyUIPromptToLoras":
        {
            MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
            MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        },
    "WeiLinPromptUI":
        {
            MetaField.LORA_MODEL_NAME: {"selector": get_lora_model_name_stack},
            MetaField.LORA_MODEL_HASH: {"selector": get_lora_model_hash_stack},
            MetaField.LORA_STRENGTH_MODEL: {"selector": get_lora_strength_model_stack},
            MetaField.LORA_STRENGTH_CLIP: {"selector": get_lora_strength_clip_stack},
            # by @Aaalice
            MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
            MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        },
}
