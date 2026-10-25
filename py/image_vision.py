"""Vision analysis pass for the H3 Image Promptor node.

Unlike the video vision pass (vision.py), this module fires a separate
engine.generate_text() call per image slot, each with a role-specific
instruction determined by the task. No single batched pass — each slot
has a different job and must be analysed independently.

Presets are read from prompts/vision_presets.json on every call so users
can tweak instructions without restarting ComfyUI.
"""

from __future__ import annotations

import os

from . import engine, vision as _vision

# ---------------------------------------------------------------------------
# System prompt for the image vision pass.
# Replaces VISION_SYSTEM_PROMPT from vision.py for image tasks.
# ---------------------------------------------------------------------------

IMAGE_VISION_SYSTEM_PROMPT = (
    "You are an expert visual analyst assisting a still-image generation prompt writer. "
    "Your job is to extract specific, factual visual information from reference images. "
    "Describe only what is actually visible. Never speculate, never invent, never infer "
    "mood, intention, or narrative. Output plain descriptive text only — no JSON, no "
    "markdown fences, no commentary about yourself or your analysis process.\n\n"

    "CRITICAL — CHARACTER SHEETS: A reference image may show multiple views, angles, "
    "or poses of the SAME individual on one sheet. Do not interpret multiple figures "
    "on a single image as multiple people unless the analysis instruction explicitly "
    "states the image contains more than one person. When in doubt, treat all figures "
    "as one individual and extract only their consistent identity features.\n\n"

    "CRITICAL — ROLE DISCIPLINE: Each image has exactly one role defined in your "
    "analysis instruction. Extract only what that role permits. If your instruction "
    "says identity, extract only identity. If it says pose, extract only pose. "
    "If it says environment, extract only environment. Never volunteer information "
    "outside your assigned role even if that information is clearly visible in the image."
)

# ---------------------------------------------------------------------------
# Task-to-slot preset mapping.
# Defines which vision preset key fires for each image slot index.
# Slot index == position in the autogrow image list (0-based).
# ---------------------------------------------------------------------------

IMAGE_TASK_SLOT_PRESETS: dict[str, dict[int, str]] = {
    "Ref2Image_1P": {
        0: "image_identity",
    },
    "Ref2Image_1P_Scene": {
        0: "image_identity",
        1: "image_environment",
    },
    "Ref2Image_1P_Pose": {
        0: "image_identity",
        1: "image_environment",
        2: "image_pose",
    },
    "Ref2Image_2P_Scene": {
        0: "image_identity",
        1: "image_identity",
        2: "image_environment",
    },
    "Ref2Image_2P_Pose": {
        0: "image_identity",
        1: "image_identity",
        2: "image_environment",
        3: "image_pose",
    },
}

# Human-readable slot role labels for debug output and warnings.
_SLOT_ROLE_LABELS: dict[str, str] = {
    "image_identity":    "IDENTITY",
    "image_environment": "ENVIRONMENT",
    "image_pose":        "POSE",
}


def _load_preset(preset_key: str) -> str:
    """Read a single preset instruction from vision_presets.json."""
    presets = _vision.load_vision_presets()
    instruction = presets.get(preset_key)
    if instruction is None:
        raise RuntimeError(
            f"[H3 ImagePromptor] Vision preset '{preset_key}' not found in "
            "vision_presets.json. Check the file contains the image_identity, "
            "image_environment, and image_pose keys."
        )
    return instruction


def _picture_label(slot_index: int) -> str:
    """Convert 0-based slot index to 1-based <Picture N> label."""
    return f"<Picture {slot_index + 1}>"


def analyze_images_for_task(
    clip,
    image_tensor,
    task: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[str]]:
    """Run per-slot role-specific vision passes for the given image task.

    Returns:
        vision_context: assembled '<Picture N> [ROLE]: description' string,
                        ready to pass to build_image_messages().
        warnings:       list of warning strings for the debug output.
    """
    if image_tensor is None:
        return "", []

    slot_presets = IMAGE_TASK_SLOT_PRESETS.get(task)
    if slot_presets is None:
        raise ValueError(
            f"[H3 ImagePromptor] No slot-preset mapping defined for task '{task}'. "
            f"Expected one of: {', '.join(IMAGE_TASK_SLOT_PRESETS.keys())}."
        )

    n_images = int(image_tensor.shape[0])
    expected = len(slot_presets)
    warnings: list[str] = []

    if n_images < expected:
        warnings.append(
            f"Task '{task}' expects {expected} image(s) but only {n_images} "
            f"supplied. Missing slots will be skipped — output may be incomplete."
        )
    if n_images > expected:
        warnings.append(
            f"Task '{task}' expects {expected} image(s) but {n_images} supplied. "
            f"Extra image(s) beyond slot {expected} will be ignored."
        )

    import torch  # only available inside ComfyUI runtime

    parts: list[str] = []

    for slot_index in range(min(n_images, expected)):
        preset_key = slot_presets[slot_index]
        role_label = _SLOT_ROLE_LABELS.get(preset_key, preset_key.upper())
        picture_label = _picture_label(slot_index)

        try:
            instruction = _load_preset(preset_key)
        except RuntimeError as e:
            warnings.append(str(e))
            parts.append(f"{picture_label} [{role_label}]: (vision analysis failed — preset not found)")
            continue

        # Inject the correct <Picture N> placeholder into identity instructions
        # so the VLM outputs the binding line with the right label.
        instruction = instruction.replace("<Subject N>", f"<Subject {slot_index + 1}>")
        instruction = instruction.replace("<Picture N>", picture_label)

        single = image_tensor[slot_index : slot_index + 1]

        try:
            desc = engine.generate_text(
                clip,
                IMAGE_VISION_SYSTEM_PROMPT,
                instruction,
                image_tensor=single,
                seed=int(seed) + slot_index,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                max_tokens=max_tokens,
            ).strip()
        except Exception as e:
            warnings.append(
                f"{picture_label} [{role_label}] vision pass failed: {e}"
            )
            desc = f"(vision analysis failed: {e})"

        parts.append(f"{picture_label} [{role_label}]:\n{desc}")

    vision_context = "\n\n".join(parts)
    return vision_context, warnings