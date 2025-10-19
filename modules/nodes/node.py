import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import piexif
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enum import Enum

import folder_paths

from .. import hook
from ..capture import Capture
from ..trace import Trace
from ..utils.log import print_warning


class OutputFormat(str, Enum):
    PNG = "png"
    PNG_JSON = "png_with_json"
    JPG = "jpg"
    JPG_JSON = "jpg_with_json"
    WEBP = "webp"
    WEBP_JSON = "webp_with_json"


class QualityOption(str, Enum):
    MAX = "max"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MetadataScope(str, Enum):
    FULL = "full"
    DEFAULT = "default"
    PARAMETERS_ONLY = "parameters_only"
    WORKFLOW_ONLY = "workflow_only"
    NONE = "none"


# refer. https://github.com/comfyanonymous/ComfyUI/blob/38b7ac6e269e6ecc5bdd6fefdfb2fb1185b09c9d/nodes.py#L1411
class SaveImageWithMetaData:
    OUTPUT_FORMATS = [e for e in OutputFormat]
    QUALITY_OPTIONS = [e for e in QualityOption]
    METADATA_OPTIONS = [e for e in MetadataScope]
    NEEDS_METADATA_KEYS = {"seed", "width", "height", "pprompt", "nprompt", "model"}

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the saved file. You can include formatting options like %date:yyyy-MM-dd% or %seed%, and combine them as needed, e.g., %date:hhmmss%_%seed%."}),
                "subdirectory_name": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom directory to save the images. Leave empty to use the default output "
                        "directory. You can include formatting options like %date:yyyy-MM-dd%."
                    ),
                }),
                "output_format": (s.OUTPUT_FORMATS, {
                    "tooltip": "The format in which the images will be saved."
                }),
            },
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {
                    "tooltip": "Additional key-value metadata to include in the image."
                }),
                "quality": (s.QUALITY_OPTIONS, {
                    "tooltip": "Image quality:"
                            "\n'max' / 'lossless WebP' - 100"
                            "\n'high' - 80"
                            "\n'medium' - 60"
                            "\n'low' - 30"
                            "\n\nNote: Lower quality, smaller file size. PNG images ignore this setting."
                }),
                "metadata_scope": (s.METADATA_OPTIONS, {
                    "tooltip": "Choose the metadata to save: "
                            "\n'full' - default metadata with additional metadata, "
                            "\n'default' - same as SaveImage node, "
                            "\n'parameters_only' - only A1111-style metadata, "
                            "\n'workflow_only' - workflow metadata only, "
                            "\n'none' - no metadata."
                }),
                "include_batch_num": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include batch number in filename."
                }),
                "prefer_nearest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Select inputs from closest nodes first if true."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the input images with metadata to your ComfyUI output directory."
    CATEGORY = "SaveImage"

    pattern_format = re.compile(r"(%[^%]+%)") # Pattern to match mask values in the filename

    def parse_output_format(self, output_format: str):
        fmt = OutputFormat(output_format)
        save_workflow_json = fmt.name.endswith("JSON")
        base_format = fmt.replace("_with_json", "")
        return base_format, save_workflow_json

    def get_quality_value(self, quality: str) -> int:
        return {
            QualityOption.MAX: 100,
            QualityOption.HIGH: 80,
            QualityOption.MEDIUM: 60,
            QualityOption.LOW: 30
        }.get(quality, 100)

    def find_next_available_filename(self, folder: str, name: str, ext: str):
        """
        Finds the next available filename by checking existing files in the directory.
        """
        existing = {f.stem for f in Path(folder).glob(f"{name}_*.{ext}")}
        i = 1
        while f"{name}_{i:05d}" in existing:
            i += 1
        return i

    @classmethod
    def parse_filename_placeholders(cls, filename: str) -> list[str]:
        """Extracts placeholder segments like %seed%, %pprompt:32%, etc."""
        return re.findall(cls.pattern_format, filename) if "%" in filename else []

    def needs_pnginfo_in_filename(self, segments: list[str]) -> bool:
        for segment in segments:
            parts = segment.strip("%").split(":")
            if parts[0] in self.NEEDS_METADATA_KEYS:
                return True
        return False

    def save_images(self, images, filename_prefix="ComfyUI", subdirectory_name="", prompt=None,
                    extra_pnginfo=None, extra_metadata=None, output_format="png",
                    quality="max", metadata_scope="full",
                    include_batch_num=True, prefer_nearest=True, pnginfo_dict=None):

        extra_metadata = extra_metadata or {}
        base_format, save_workflow_json = self.parse_output_format(output_format)
        pnginfo = PngInfo()

        # Parse filename
        filename_prefix = filename_prefix.strip()
        segments = self.parse_filename_placeholders(filename_prefix)

        if metadata_scope in [MetadataScope.FULL, MetadataScope.PARAMETERS_ONLY] or self.needs_pnginfo_in_filename(segments):
            pnginfo_dict = pnginfo_dict or self.gen_pnginfo(prompt, prefer_nearest)

        filename_prefix = self.format_filename(filename_prefix, pnginfo_dict or {}, segments) + self.prefix_append
        subdirectory_name = self.format_filename(subdirectory_name, pnginfo_dict or {})


        image_shape = images[0].shape
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, image_shape[1], image_shape[0]
        )

        # Handle subdirectory naming and creation
        subdirectory_name = subdirectory_name.strip()
        if subdirectory_name:
            subdirectory_name = self.format_filename(subdirectory_name, pnginfo_dict)
            full_output_folder = os.path.join(self.output_dir, subdirectory_name)
            filename = filename_prefix

        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        images_length = len(images)
        last_image_filename = None

        # Process each image
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Prepare metadata
            metadata = self.prepare_pnginfo(pnginfo, pnginfo_dict, batch_number, images_length, prompt, extra_pnginfo, metadata_scope)
            for key, value in extra_metadata.items():
                metadata.add_text(key, value)

            # Handle filename collision and batch number inclusion
            file = f"{filename}_{batch_number:05d}.{base_format}" if include_batch_num else f"{filename}.{base_format}"
            path = os.path.join(full_output_folder, file)

            # Check for filename collision (using next available name)
            if os.path.exists(path):
                count = self.find_next_available_filename(full_output_folder, filename, base_format)
                file = f"{filename}_{count:05d}.{base_format}"
                path = os.path.join(full_output_folder, file)

            last_image_filename = file
            quality_value = self.get_quality_value(quality)

            # Save image based on format
            if base_format == "webp":
                img.save(path, "WEBP", lossless=(quality_value == 100), quality=quality_value)
            elif base_format == "png":
                img.save(path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(path, optimize=True, quality=quality_value)

            # Insert EXIF for jpg/webp formats
            if base_format in ["jpg", "webp"]:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(Capture.gen_parameters_str(pnginfo_dict), encoding="unicode")
                    }
                })
                piexif.insert(exif_bytes, path)

            results.append({"filename": file, "subfolder": full_output_folder, "type": self.type})

        # Save workflow metadata for the batch
        if save_workflow_json and images_length > 0 and last_image_filename:
            json_filename = last_image_filename.replace(base_format, "json")
            batch_json_file = os.path.join(full_output_folder, json_filename)

            with open(batch_json_file, "w", encoding="utf-8") as f:
                json.dump(extra_pnginfo["workflow"], f)

        return {"ui": {"images": results}}

    def prepare_pnginfo(self, metadata, pnginfo_dict, batch_number, total_images, prompt, extra_pnginfo, metadata_scope):
        """
        Return final PNG metadata with batch information, parameters, and optional prompt details.
        """
        if metadata_scope == MetadataScope.NONE:
            return None

        if pnginfo_dict:
            pnginfo_copy = pnginfo_dict.copy()

            if total_images > 1:
                pnginfo_copy["Batch index"] = batch_number
                pnginfo_copy["Batch size"] = total_images

            if metadata_scope in [MetadataScope.FULL, MetadataScope.PARAMETERS_ONLY]:
                parameters = Capture.gen_parameters_str(pnginfo_copy)
                if parameters and "Steps" in parameters:
                    metadata.add_text("parameters", parameters)
                    if metadata_scope == MetadataScope.PARAMETERS_ONLY:
                        return metadata

        if prompt is not None and metadata_scope != MetadataScope.WORKFLOW_ONLY:
            metadata.add_text("prompt", json.dumps(prompt))

        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        return metadata

    @classmethod
    def gen_pnginfo(self, prompt, prefer_nearest=None):
        from ..capture import Capture
        from collections import defaultdict
        
        # This is the robust way to collect metadata by analyzing the prompt,
        # which is compatible with the new ComfyUI version. It bypasses the
        # broken Capture.get_inputs() method entirely.
        inputs_before_sampler_node = defaultdict(list)
        Capture._collect_all_metadata(prompt, inputs_before_sampler_node)
        
        # For this node's logic, the metadata up to the sampler is the same
        # as the metadata up to the save node itself.
        inputs_before_this_node = inputs_before_sampler_node

        return Capture.gen_pnginfo_dict(
            inputs_before_sampler_node, inputs_before_this_node, prompt
        )

    @classmethod
    def format_filename(cls, filename, pnginfo_dict, segments=None):
        """
        Replaces placeholders in the filename with actual values like date, seed, prompt, etc.
        """
        if "%" not in filename:
            return filename

        segments = segments or re.findall(cls.pattern_format, filename)
        now = datetime.now()
        date_table = {
            "yyyy": f"{now.year}",
            "MM": f"{now.month:02d}",
            "dd": f"{now.day:02d}",
            "hh": f"{now.hour:02d}",
            "mm": f"{now.minute:02d}",
            "ss": f"{now.second:02d}",
        }

        for segment in segments:
            parts = segment.strip("%").split(":")
            key = parts[0]

            if key == "seed":
                seed = pnginfo_dict.get("Seed")
                if seed is None:
                    print_warning("Seed not found in pnginfo_dict!")
                filename = filename.replace(segment, str(seed or ""))

            elif key in {"width", "height"}:
                size = pnginfo_dict.get("Size", "x").split("x")
                if "Size" not in pnginfo_dict:
                    print_warning("Size not found in pnginfo_dict!")
                value = size[0] if key == "width" else size[1]
                filename = filename.replace(segment, value)

            elif key in {"pprompt", "nprompt"}:
                prompt_key = "Positive prompt" if key == "pprompt" else "Negative prompt"
                prompt = pnginfo_dict.get(prompt_key, "")
                if not prompt:
                    print_warning(f"{prompt_key} not found in pnginfo_dict!")
                prompt = prompt.replace("\n", " ")
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, prompt[:length].strip() if length else prompt.strip())

            elif key == "model":
                model = pnginfo_dict.get("Model", "")
                if not model:
                    print_warning("Model not found in pnginfo_dict!")
                model = os.path.splitext(os.path.basename(model))[0]
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, model[:length] if length else model)

            elif key == "date":
                date_format = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename = filename.replace(segment, date_format)

        return filename


class CreateExtraMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {"forceInput": True}),
                **{
                    f"{type}{i}": ("STRING", {"default": "", "multiline": False})
                    for i in range(1, 5)
                    for type in ["key", "value"]
                },
            }
        }

    RETURN_TYPES = ("EXTRA_METADATA",)
    FUNCTION = "create_extra_metadata"
    DESCRIPTION = "Creates custom extra metadata by adding key-value pairs. Empty values are allowed, but unpaired values are not."
    CATEGORY = "SaveImage"

    def create_extra_metadata(self, extra_metadata=None, **keys_values):
        if extra_metadata is None:
            extra_metadata = {}

        for i in range(1, 5):
            key = keys_values.get(f"key{i}", "").strip()
            value = keys_values.get(f"value{i}", "").strip()

            if key:
                extra_metadata[key] = value
            elif value:
                raise ValueError(f"Value provided for 'value{i}' without corresponding 'key{i}'.")

        return (extra_metadata,)
