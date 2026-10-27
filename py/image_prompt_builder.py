"""Prompt assembly for the H3 Image Promptor node.

Separate from prompt_builder.py — no duration, no frames, no video language,
no audio fields. Image tasks only.
"""

from __future__ import annotations

import os
import warnings

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
_IDEA_WARN_THRESHOLD = 30

IMAGE_TASK_TYPES = (
    "Ref2Image_1P",
    "Ref2Image_1P_Pose",
    "Ref2Image_1P_Scene",
    "Ref2Image_2P_Scene",
    "Ref2Image_2P_Pose",
)

def load_image_system_prompt(task: str, custom_system_prompt: str = "") -> str:
    if custom_system_prompt and custom_system_prompt.strip():
        return custom_system_prompt.strip()
    
    # Load base system prompt
    base_file = os.path.join(_PROMPTS_DIR, "hs3_image_base.txt")
    if os.path.exists(base_file):
        with open(base_file, "r", encoding="utf-8") as f:
            base = f.read().strip()
    else:
        base = f"You are an expert image prompt writer for the {task} task."
    
    # Load and append task-specific addendum
    task_file = os.path.join(_PROMPTS_DIR, "tasks", f"{task.lower()}.txt")
    if os.path.exists(task_file):
        with open(task_file, "r", encoding="utf-8") as f:
            addendum = f.read().strip()
        return base + "\n\n" + addendum
    
    return base

def detect_image_task(task_type: str, n_images: int) -> str:
    t = (task_type or "Auto").strip()
    if t.lower() != "auto":
        for known in IMAGE_TASK_TYPES:
            if t.lower() == known.lower():
                return known
        raise ValueError(
            f"[H3 ImagePromptor] Unknown task_type '{task_type}'. "
            f"Expected one of Auto, {', '.join(IMAGE_TASK_TYPES)}."
        )
    if n_images <= 1:
        return "Ref2Image_1P"
    if n_images == 2:
        return "Ref2Image_1P_Scene"
    if n_images == 3:
        return "Ref2Image_1P_Pose"
    return "Ref2Image_2P_Pose"


def word_budget(task: str) -> int:
    if task in ("Ref2Image_2P_Scene", "Ref2Image_2P_Pose"):
        return 250
    return 200

def build_image_messages(
    task: str,
    user_idea: str,
    vision_context: str,
    extra_instructions: str,
    custom_system_prompt: str = "",
) -> tuple[str, str]:
    """Build the (system, user) pair for the image prompt-writing generation."""

    system = load_image_system_prompt(task, custom_system_prompt)
    budget = word_budget(task)

    user_parts: list[str] = []

    # Vision context from the analysis pass
    if vision_context and vision_context.strip():
        user_parts.append(
            "Reference image analysis (facts about the provided picture(s); "
            "<Picture N> refers to the Nth provided image in order):\n"
            + vision_context.strip()
        )

    # User idea — warn if thin, especially for pose tasks
    idea = (user_idea or "").strip()
    if idea:
        if len(idea) < _IDEA_WARN_THRESHOLD:
            warnings.warn(
                f"[H3 ImagePromptor] user_idea is very short ({len(idea)} chars). "
                "For pose-reference tasks a thin idea risks the model anchoring "
                "to the pose image's scenario rather than the user's intent. "
                "Consider adding: who, where, and what they are doing.",
                stacklevel=2,
            )
        user_parts.append("User idea:\n" + idea)
    else:
        if "pose" in task.lower():
            user_parts.append(
                "User idea: (none provided)\n"
                "WARNING: no user idea supplied for a pose-reference task. "
                "The model will attempt to infer a scenario from the reference "
                "images. This is unreliable — the pose reference may dominate "
                "and its scenario will bleed into the output. "
                "Strongly recommend supplying at least: who, where, what they are doing."
            )
        else:
            user_parts.append(
                "User idea: (none provided — invent a coherent, concrete scene "
                "consistent with the reference image analysis, if any.)"
            )

    # Word budget instruction — no duration, no frames
    user_parts.append(
        f"Write the image_description field at roughly {budget} words "
        f"(target range {budget - 50}–{budget + 50} words). "
        "Prioritise concrete, complete visual description over hitting the count exactly. "
        "This is a single still frame — no motion, no timeline, no sequence."
    )

    # Extra instructions
    if extra_instructions and extra_instructions.strip():
        user_parts.append(
            "Additional instructions from the user (respect them unless they "
            "conflict with the image format rules):\n" + extra_instructions.strip()
        )

    # Language
    user_parts.append(
        "Language: write the prompt body in English. "
        "On-screen text keeps its original language verbatim."
    )

    return system, "\n\n".join(user_parts)