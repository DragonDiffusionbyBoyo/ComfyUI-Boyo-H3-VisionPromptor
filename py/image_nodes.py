"""H3 Image Promptor node — still-image prompt builder using a local VLM.

Separate from nodes.py to keep image and video node definitions isolated.
Shares engine, vision machinery, and helper functions with the video nodes.
"""

from __future__ import annotations

import json
import time

import folder_paths
from comfy_api.latest import io

from . import engine
from . import image_prompt_builder
from . import image_vision

CATEGORY = "H3/Promptor"
MAX_SEED = 0xFFFFFFFF

IMAGE_TASK_OPTIONS = [
    "Auto",
    "Ref2Image_1P",
    "Ref2Image_1P_Pose",
    "Ref2Image_1P_Scene",
    "Ref2Image_2P_Scene",
    "Ref2Image_2P_Pose",
]

def _text_encoder_options() -> list[str]:
    try:
        return [NONE_OPTION] + folder_paths.get_filename_list("text_encoders")
    except Exception:
        return [NONE_OPTION]

NONE_OPTION = "<none - use CLIP input>"


def _images_autogrow() -> io.Autogrow.Input:
    return io.Autogrow.Input(
        "images",
        optional=True,
        tooltip="Reference images. Slot roles are fixed by task: "
                "Ref2Image_1P_Pose = image_0:identity, image_1:environment, image_2:pose",
        template=io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image", tooltip="Reference image"),
            prefix="image_",
            min=0,
            max=4,
        ),
    )


def _collect_images(images):
    if images is None:
        return None
    values = []
    if isinstance(images, dict):
        for key in sorted(images.keys()):
            values.append(images[key])
    elif isinstance(images, (list, tuple)):
        values = list(images)
    else:
        values = [images]
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    import torch
    min_h = min(int(v.shape[1]) for v in values)
    min_w = min(int(v.shape[2]) for v in values)
    cropped = []
    for v in values:
        h, w = int(v.shape[1]), int(v.shape[2])
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        cropped.append(v[:, top:top + min_h, left:left + min_w, :])
    return torch.cat(cropped, dim=0)


def _n_images(image_tensor) -> int:
    if image_tensor is None:
        return 0
    try:
        return int(image_tensor.shape[0])
    except Exception:
        return 1


class H3ImagePromptor(io.ComfyNode):
    """Local, API-key-free still-image prompt builder powered by a native VLM.

    Produces structured reference-image prompts for AI image generation models.
    No duration, no timeline, no audio. Each reference image slot has a fixed
    role determined by the selected task.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ImagePromptor",
            display_name="H3 Image Promptor (Local VLM)",
            category=CATEGORY,
            search_aliases=["image", "still", "photo", "reference", "VLM", "prompt"],
            description=(
                "Turns an idea (+ reference images) into a production-ready still-image "
                "prompt using a LOCAL VLM. Each image slot has a fixed role: identity, "
                "environment, or pose — determined automatically by the selected task. "
                "No API keys, no external services."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: connect CLIP from the native CLIPLoader node. "
                                      "Overrides the text_encoder dropdown."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/."),
                io.String.Input("user_idea", multiline=True, default="",
                                placeholder=(
                                    "Who is in the image, where, and what are they doing? "
                                    "e.g. '<Subject 1> from <Picture 1> stands in the park "
                                    "shown in <Picture 2>, in the pose shown in <Picture 3>'"
                                )),
                _images_autogrow(),
                io.Combo.Input("task_type", options=IMAGE_TASK_OPTIONS, default="Auto",
                               tooltip=(
                                   "Auto: 1 image=Ref2Image_1P, 2-3=Ref2Image_1P_Pose, "
                                   "4=Ref2Image_2P_Pose. Select explicitly to override."
                               )),
                io.String.Input("extra_instructions", multiline=True, default="",
                                optional=True,
                                placeholder=(
                                    "Optional: clothing details, style direction, "
                                    "lighting preference, specific composition notes..."
                                )),
                io.String.Input("custom_system_prompt", multiline=True, default="",
                                optional=True, advanced=True,
                                tooltip="Advanced: fully replace the built-in image system prompt."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=64, min=0, max=500),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64),
                io.Int.Input("variants", default=1, min=1, max=4,
                             tooltip="Generate N prompt variants in one run."),
                io.Boolean.Input("use_default_template", default=False, advanced=True,
                                 tooltip="Let the model's built-in chat template wrap the prompt "
                                         "instead of this node's image-aware template."),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs."),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("vision_context"),
                io.String.Output("debug"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, user_idea, task_type,
                seed, temperature, top_p, top_k, max_tokens, variants,
                use_default_template, keep_model_loaded,
                clip=None, images=None, extra_instructions="",
                custom_system_prompt="") -> io.NodeOutput:

        t_start = time.time()

        clip_obj, source_desc = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        image_tensor = _collect_images(images)
        n_images = _n_images(image_tensor)

        task = image_prompt_builder.detect_image_task(task_type, n_images)

        # --- vision pass (per-slot, role-specific) ------------------------
        vision_context = ""
        vision_seconds = 0.0
        vision_warnings: list[str] = []

        if image_tensor is not None and n_images > 0:
            t_vis = time.time()
            try:
                vision_context, vision_warnings = image_vision.analyze_images_for_task(
                    clip_obj,
                    image_tensor,
                    task,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                vision_warnings.append(f"vision pass failed entirely: {e}")
                print(f"[H3 ImagePromptor] Vision pass failed, continuing without it: {e}")
            vision_seconds = time.time() - t_vis

        # --- prompt-writing generation ------------------------------------
        system, user = image_prompt_builder.build_image_messages(
            task=task,
            user_idea=user_idea,
            vision_context=vision_context,
            extra_instructions=extra_instructions,
            custom_system_prompt=custom_system_prompt,
        )

        results: list[str] = []
        all_warnings: list[str] = list(vision_warnings)
        t_gen = time.time()

        for i in range(int(variants)):
            raw = engine.generate_text(
                clip_obj, system, user,
                image_tensor=None,
                seed=int(seed) + i,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                use_default_template=use_default_template,
            )
            results.append(raw.strip())

        gen_seconds = time.time() - t_gen

        prompt = results[0]
        for i in range(1, len(results)):
            prompt += f"\n\n===== VARIANT {i + 1} =====\n\n" + results[i]

        debug = json.dumps({
            "clip_source": source_desc,
            "model_family": engine.detect_family(clip_obj),
            "detected_task": task,
            "n_images": n_images,
            "word_budget": image_prompt_builder.word_budget(task),
            "vision_seconds": round(vision_seconds, 2),
            "generation_seconds": round(gen_seconds, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "variants": int(variants),
            "warnings": all_warnings,
        }, indent=2, ensure_ascii=False)

        return io.NodeOutput(prompt, vision_context, debug)