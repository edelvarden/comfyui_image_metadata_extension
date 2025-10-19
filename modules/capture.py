import json
import os
import re
from collections import defaultdict
#from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField
from .defs.formatters import calc_lora_hash, calc_model_hash, extract_embedding_names, extract_embedding_hashes
from .utils.log import print_warning

from nodes import NODE_CLASS_MAPPINGS
from .trace import Trace
from execution import get_input_data
from comfy_execution.graph import DynamicPrompt, ExecutionList 


class Capture:
    @classmethod
    def get_inputs(self):
        if Capture.instance is None:
            return {}

        prompt_executor = execution.PromptExecutor.instance
        if not hasattr(prompt_executor, 'prompt'):
            return {}

        # Get the necessary objects from the current execution state
        prompt = prompt_executor.prompt
        caches = prompt_executor.caches
        dynamic_prompt = DynamicPrompt(prompt)
        execution_list = ExecutionList(dynamic_prompt, caches.outputs)

        inputs = {}
        for node_id, (node, active) in Capture.instance.nodes.items():
            if not active:
                continue

            # Ensure the node exists in the current prompt before processing
            if node_id not in prompt:
                continue
            
            class_def = nodes.NODE_CLASS_MAPPINGS[node["class_type"]]
            
            # Call get_input_data with the correct arguments: execution_list and dynamic_prompt
            input_data, _, _ = get_input_data(
                node["inputs"],
                class_def,
                node_id,
                execution_list, # Pass the ExecutionList object
                dynamic_prompt, # Pass the DynamicPrompt object
            )

            if not input_data:
                continue

            for name, data in input_data.items():
                input_info = next(
                    (
                        i
                        for i in node["inputs"]
                        if i.get("name") == name and type(i.get("link")) is int
                    ),
                    None,
                )
                if input_info is not None:
                    continue

                if name in inputs and inputs[name] != data[0]:
                    print(
                        f"Conflicting input values for '{name}': {inputs[name]} vs {data[0]}"
                    )
                inputs[name] = data[0]

        return inputs
    
    @staticmethod
    def _apply_formatting(value, input_data, format_func):
        """Apply formatting to a value using the given format function."""
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if format_func:
            value = format_func(value, input_data)
        return value

    @staticmethod
    def _append_value(inputs, meta, node_id, value):
        """Append processed value to the inputs list."""
        if isinstance(value, list):
            for x in value:
                inputs[meta].append((node_id, x))
        elif value is not None:
            inputs[meta].append((node_id, value))

    @classmethod
    def get_lora_strings_and_hashes(cls, inputs_before_sampler_node):

        def clean_name(n):
            return os.path.splitext(os.path.basename(n))[0].replace('\\', '_').replace('/', '_').replace(' ', '_').replace(':', '_')

        # Regex to match <lora:name:weight>, based on https://github.com/civitai/civitai/blob/main/src/utils/prompt-helpers.ts
        lora_assertion_re = re.compile(r"<(lora|lyco):([a-zA-Z0-9_\./\\-]+):([0-9.]+)>")

        prompt_texts = [
            val[1]
            for key in [MetaField.POSITIVE_PROMPT, MetaField.NEGATIVE_PROMPT]
            for val in inputs_before_sampler_node.get(key, [])
            if isinstance(val[1], str)
        ]
        prompt_joined = " ".join(prompt_texts).replace("\n", " ").replace("\r", " ").lower()

        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_weights = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])
        lora_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])

        # Parse LoRAs in prompt
        lora_names_from_prompt, lora_weights_from_prompt, lora_hashes_from_prompt = [], [], []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                for _, name, weight in re.findall(lora_assertion_re, text.replace("\n", " ").replace("\r", " ")):
                    lora_names_from_prompt.append(("prompt_parse", name))
                    lora_weights_from_prompt.append(("prompt_parse", float(weight)))

                    h = calc_lora_hash(name)
                    if h:
                        lora_hashes_from_prompt.append(("prompt_parse", h))

        # Combine all sources
        all_names = lora_names + lora_names_from_prompt
        all_weights = lora_weights + lora_weights_from_prompt
        all_hashes = lora_hashes + lora_hashes_from_prompt

        # Update the metadata fields with combined information
        inputs_before_sampler_node[MetaField.LORA_MODEL_NAME] = all_names
        inputs_before_sampler_node[MetaField.LORA_STRENGTH_MODEL] = all_weights
        inputs_before_sampler_node[MetaField.LORA_MODEL_HASH] = all_hashes

        grouped = defaultdict(list)
        for name, weight, hsh in zip(all_names, all_weights, all_hashes):
            if not (name and weight and hsh):
                continue
            grouped[(hsh[1], weight[1])].append(clean_name(name[1]))

        hashes_in_prompt = {h[1].lower() for h in lora_hashes_from_prompt}

        lora_strings, lora_hashes_list = [], []

        for (hsh, weight), names in grouped.items():
            canonical = min(names, key=len)
            present = hsh.lower() in hashes_in_prompt

            if not present:
                lora_strings.append(f"<lora:{canonical}:{weight}>")
            lora_hashes_list.append(f"{canonical}: {hsh}")

        # Rewrite prompt with cleaned names
        updated_prompts = []
        if "<lora:" in prompt_joined:
            for text in prompt_texts:
                def replace(m):
                    tag, raw_name, weight = m.group(1), m.group(2), m.group(3)
                    return f"<{tag}:{clean_name(raw_name)}:{weight}>"
                updated_prompts.append(lora_assertion_re.sub(replace, text))
        else:
            updated_prompts = prompt_texts

        lora_hashes_string = ", ".join(lora_hashes_list)
        return lora_strings, lora_hashes_string, updated_prompts

    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, prompt, save_civitai_sampler=True):
        pnginfo = {}

        if not inputs_before_sampler_node:
            inputs_before_sampler_node = defaultdict(list)
            cls._collect_all_metadata(prompt, inputs_before_sampler_node)

        def is_simple(value):
            return isinstance(value, (str, int, float, bool)) or value is None
        
        def extract(meta_key, label, source=inputs_before_sampler_node):
            """
            Scan the list behind `meta_key` and return the first payload that:
              1) is present (link has at least two elements),
              2) is not None,
              3) if it's a string, the string is not empty.
            Once found, it stores the value in `pnginfo[label]` and exits early.
            """
            # Retrieve the list associated with `meta_key`; default to an empty list
            val_list = source.get(meta_key, [])
            # Traverse the list in the original order (front to back)
            for link in val_list:
                # Guard against malformed link entries with length < 2
                if len(link) <= 1:
                    continue
                candidate = link[1]
                # Skip if None
                if candidate is None:
                    continue
                # If candidate is a string, skip empty ones
                if isinstance(candidate, str):
                    if not candidate.strip():
                        continue
                elif not is_simple(candidate):
                    continue # Skip lists, dicts, etc.

                value = str(candidate)
                pnginfo[label] = value
                return value

            # No valid payload found
            return None

        # Prompts
        positive = extract(MetaField.POSITIVE_PROMPT, "Positive prompt") or ""
        if not positive.strip():
            print_warning("Positive prompt is empty!")

        negative = extract(MetaField.NEGATIVE_PROMPT, "Negative prompt") or ""
        lora_strings, lora_hashes, updated_prompts = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)
        
        # If there are LoRAs in the prompt, use the cleaned version of the prompt.
        if updated_prompts:
            positive = updated_prompts[0]
            
        # Append LoRA models to the positive prompt, which is required for the Civitai website to parse and apply LoRA weights.
        # Format: <lora:Lora_Model_Name:weight_value>. Example: <lora:Lora_Name_00:0.6> <lora:Lora_Name_01:0.8>
        if lora_strings:
            positive += " " + " ".join(lora_strings)

        pnginfo["Positive prompt"] = positive.strip()
        pnginfo["Negative prompt"] = negative.strip()

        # Sampling params
        if not extract(MetaField.STEPS, "Steps"):
            print_warning("Steps are empty, full metadata won't be added!")
            return {}  # No sense in pnginfo without the Steps parameter, ref https://github.com/civitai/civitai/blob/7c8f3f3044218cf3b3d86bd9f49d12fc196ea1f6/src/utils/metadata/automatic.metadata.ts#L102C42-L102C47

        samplers = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME)
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER)

        if save_civitai_sampler:
            pnginfo["Sampler"] = cls.get_sampler_for_civitai(samplers, schedulers)
        elif samplers:
            sampler_name = samplers[0][1]
            if schedulers and schedulers[0][1] != "normal":
                sampler_name += f"_{schedulers[0][1]}"
            pnginfo["Sampler"] = sampler_name

        extract(MetaField.CFG, "CFG scale")
        extract(MetaField.SEED, "Seed")
        
        # Missing CLIP skip means it was set to 1 (the default)
        clip_skip = extract(MetaField.CLIP_SKIP, "Clip skip")
        if clip_skip is None:
            pnginfo["Clip skip"] = "1"

        # Image size
        image_width_data = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [[None]])
        image_height_data = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [[None]])

        def extract_dimension(data):
            return data[0][1] if data and len(data[0]) > 1 and isinstance(data[0][1], int) else None

        width = extract_dimension(image_width_data)
        height = extract_dimension(image_height_data)
        if width and height:
            pnginfo["Size"] = f"{width}x{height}"

        # Model details
        extract(MetaField.MODEL_NAME, "Model")
        extract(MetaField.MODEL_HASH, "Model hash")
        extract(MetaField.VAE_NAME, "VAE", inputs_before_this_node)
        extract(MetaField.VAE_HASH, "VAE hash", inputs_before_this_node)

        # Denoising strength
        denoise = inputs_before_sampler_node.get(MetaField.DENOISE)
        dval = denoise[0][1] if denoise else None
        if dval and 0 < float(dval) < 1:
            pnginfo["Denoising strength"] = float(dval)

        # Include upscale info if present
        if inputs_before_this_node.get(MetaField.UPSCALE_BY) or inputs_before_this_node.get(MetaField.UPSCALE_MODEL_NAME):
            pnginfo["Denoising strength"] = float(dval or 1.0)

        # Hi-Res, based on https://github.com/civitai/civitai/blob/0c6a61b2d3ee341e77a357d4c08cf220e22b1190/src/server/common/model-helpers.ts#L33
        extract(MetaField.UPSCALE_BY, "Hires upscale", inputs_before_this_node)
        extract(MetaField.UPSCALE_MODEL_NAME, "Hires upscaler", inputs_before_this_node)

        # Lora hashes, based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/scripts/lora_script.py#L78
        if lora_hashes:
            pnginfo["Lora hashes"] = f'"{lora_hashes}"'

        # Additional metadata
        pnginfo.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo.update(cls.gen_embeddings(inputs_before_sampler_node))

        hashes = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)
        if hashes:
            pnginfo["Hashes"] = json.dumps(hashes)

        return pnginfo

    @classmethod
    def _collect_all_metadata(cls, prompt, result_dict):
        def _append_metadata(meta, node_id, value):
            if value is not None:
                result_dict[meta].append((node_id, value, 0))

        # Detect nodes with specific fields
        resolved = {
            "prompt": Trace.find_node_with_fields(prompt, {"positive", "negative"}),
            "denoise": Trace.find_node_with_fields(prompt, {"denoise"}),
            "sampler": Trace.find_node_with_fields(prompt, {"seed", "steps", "cfg", "sampler_name", "scheduler"}),
            "size": Trace.find_node_with_fields(prompt, {"width", "height"}),
            "model": Trace.find_node_with_fields(prompt, {"ckpt_name"}),
        }

        # LoRA metadata (multiple)
        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"lora_name", "strength_model"}):
            if node is not None:
                inputs = node.get("inputs", {})
                name = inputs.get("lora_name")
                strength = inputs.get("strength_model")
                _append_metadata(MetaField.LORA_MODEL_NAME, node_id, name)
                _append_metadata(MetaField.LORA_MODEL_HASH, node_id, calc_lora_hash(name) if name else None)
                _append_metadata(MetaField.LORA_STRENGTH_MODEL, node_id, strength)

        # Model metadata
        model_node = resolved.get("model")
        if model_node and model_node[1] is not None:
            node_id, node = model_node
            inputs = node.get("inputs", {})
            name = inputs.get("ckpt_name")
            _append_metadata(MetaField.MODEL_NAME, node_id, name)
            _append_metadata(MetaField.MODEL_HASH, node_id, calc_model_hash(name) if name else None)

        # Denoise
        denoise_node = resolved.get("denoise")
        if denoise_node and denoise_node[1] is not None:
            node_id, node = denoise_node
            val = node.get("inputs", {}).get("denoise")
            _append_metadata(MetaField.DENOISE, node_id, val)

        # Sampler fields
        sampler_node = resolved.get("sampler")
        if sampler_node and sampler_node[1] is not None:
            node_id, node = sampler_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "sampler_name": MetaField.SAMPLER_NAME,
                "scheduler": MetaField.SCHEDULER,
                "seed": MetaField.SEED,
                "steps": MetaField.STEPS,
                "cfg": MetaField.CFG,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # Image size fields
        size_node = resolved.get("size")
        if size_node and size_node[1] is not None:
            node_id, node = size_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "width": MetaField.IMAGE_WIDTH,
                "height": MetaField.IMAGE_HEIGHT,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # Prompt fields
        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"positive", "negative"}):
            if node is not None:
                inputs = node.get("inputs", {})
                pos_ref = inputs.get("positive", [None])[0]
                neg_ref = inputs.get("negative", [None])[0]

                def resolve_text(ref):
                    if isinstance(ref, list): ref = ref[0]
                    if not isinstance(ref, str): return None
                    node = prompt.get(ref)
                    return node.get("inputs", {}).get("text") if node else None

                pos_text = resolve_text(pos_ref)
                neg_text = resolve_text(neg_ref)
                
                # Append positive and negative prompts
                _append_metadata(MetaField.POSITIVE_PROMPT, pos_ref, pos_text)
                _append_metadata(MetaField.NEGATIVE_PROMPT, neg_ref, neg_text)

                # Add embedding metadata collection
                if pos_text:
                    embedding_names = extract_embedding_names(pos_text)
                    embedding_hashes = extract_embedding_hashes(pos_text)
                    for name, hash_ in zip(embedding_names, embedding_hashes):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, hash_)

                if neg_text:
                    embedding_names = extract_embedding_names(neg_text)
                    embedding_hashes = extract_embedding_hashes(neg_text)
                    for name, hash_ in zip(embedding_names, embedding_hashes):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, hash_)

    @classmethod
    def extract_model_info(cls, inputs, meta_field_name, prefix):
        model_info_dict = {}
        model_names = inputs.get(meta_field_name, [])
        model_hashes = inputs.get(f"{meta_field_name}_HASH", [])

        for index, (model_name, model_hash) in enumerate(zip(model_names, model_hashes)):
            field_prefix = f"{prefix}_{index}"
            model_info_dict[f"{field_prefix} name"] = os.path.splitext(os.path.basename(model_name[1]))[0]
            model_info_dict[f"{field_prefix} hash"] = model_hash[1]

        return model_info_dict

    @classmethod
    def gen_loras(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.LORA_MODEL_NAME, "Lora")

    @classmethod
    def gen_embeddings(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.EMBEDDING_NAME, "Embedding")

    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        if not pnginfo_dict or not isinstance(pnginfo_dict, dict):
            return ""

        def clean_value(value):
            if value is None:
                return ""
            return str(value).strip().replace("\n", " ")

        def strip_embedding_prefix(text):
            return text.replace("embedding:", "")

        cleaned_dict = {k: clean_value(v) for k, v in pnginfo_dict.items()}

        pos = strip_embedding_prefix(cleaned_dict.get("Positive prompt", ""))
        neg = strip_embedding_prefix(cleaned_dict.get("Negative prompt", ""))

        result = [pos]
        if neg:
            result.append(f"Negative prompt: {neg}")

        s_list = [
            f"{k}: {v}"
            for k, v in cleaned_dict.items()
            if k not in {"Positive prompt", "Negative prompt"} and v not in {None, ""}
        ]

        result.append(", ".join(s_list))
        return "\n".join(result)

    @classmethod
    def get_hashes_for_civitai(cls, inputs_before_sampler_node, inputs_before_this_node):
        def extract_single(inputs, key):
            items = inputs.get(key, [])
            return items[0][1] if items and len(items[0]) > 1 else None

        def extract_named_hashes(names, hashes, prefix):
            result = {}
            for name, h in zip(names, hashes):
                base_name = os.path.splitext(os.path.basename(name[1]))[0]
                result[f"{prefix}:{base_name}"] = h[1]
            return result

        resource_hashes = {}

        model = extract_single(inputs_before_sampler_node, MetaField.MODEL_HASH)
        if model:
            resource_hashes["model"] = model

        vae = extract_single(inputs_before_this_node, MetaField.VAE_HASH)
        if vae:
            resource_hashes["vae"] = vae
            
        upscaler_hash = extract_single(inputs_before_this_node, MetaField.UPSCALE_MODEL_HASH)
        if upscaler_hash:
            resource_hashes["upscaler"] = upscaler_hash

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, []),
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, []),
            "lora"
        ))

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.EMBEDDING_NAME, []),
            inputs_before_sampler_node.get(MetaField.EMBEDDING_HASH, []),
            "embed"
        ))

        return resource_hashes

    @classmethod
    def get_sampler_for_civitai(cls, sampler_names, schedulers):
        """
        Get the pretty sampler name for Civitai in the form of `<Sampler Name> <Scheduler name>`.
            - `dpmpp_2m` and `karras` will return `DPM++ 2M Karras`
        
        If there is a matching sampler name but no matching scheduler name, return only the matching sampler name.
            - `dpmpp_2m` and `exponential` will return only `DPM++ 2M`

        if there is no matching sampler and scheduler name, return `<sampler_name>_<scheduler_name>`
            - `ipndm` and `normal` will return `ipndm`
            - `ipndm` and `karras` will return `ipndm_karras`

        Reference: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts
        """

        # Sampler map: https://github.com/civitai/civitai/blob/fe76d9a4406d0c7b6f91f7640c50f0a8fa1b9f35/src/server/common/constants.ts#L699
        sampler_dict = {
            'euler': 'Euler',
            'euler_ancestral': 'Euler a',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'lms': 'LMS',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_sde_gpu': 'DPM++ SDE',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
            
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc': 'UniPC',
            'uni_pc_bh2': 'UniPC',
            'lcm': 'LCM'
        }

        sampler = None
        scheduler = None
        
        # Get the sampler and scheduler values
        if sampler_names:
            if len(sampler_names) > 0:
                sampler = sampler_names[0][1]
                
        if schedulers:
            if len(schedulers) > 0:
                scheduler = schedulers[0][1]

        def get_scheduler_name(sampler_name, scheduler):
            if scheduler == "karras":
                return f"{sampler_name} Karras"
            elif scheduler == "exponential":
                return f"{sampler_name} Exponential"
            elif scheduler == "normal":
                return sampler_name
            else:
                return f"{sampler_name}_{scheduler}"

        if not sampler:
            return None
    
        if sampler in sampler_dict:
            return get_scheduler_name(sampler_dict[sampler], scheduler)

        # If no match in the dictionary, return the sampler name with scheduler appended
        return get_scheduler_name(sampler, scheduler)
