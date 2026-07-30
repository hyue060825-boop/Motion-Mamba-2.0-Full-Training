"""Build a deterministic text-to-motion prompt bank for ARDY-Mamba distillation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


BASE_ACTIONS = [
    "walks forward",
    "walks backward",
    "walks sideways",
    "walks in a circle",
    "runs forward",
    "jogs in place",
    "marches in place",
    "jumps",
    "jumps twice",
    "hops on one foot",
    "squats down",
    "stands up",
    "bends down",
    "kneels briefly",
    "turns around",
    "spins in place",
    "waves",
    "claps their hands",
    "raises both arms",
    "raises the right hand",
    "raises the left hand",
    "kicks with the right leg",
    "kicks with the left leg",
    "dances happily",
    "shuffles forward",
    "shuffles backward",
    "leans forward",
    "leans backward",
    "reaches forward",
    "reaches to the left",
    "reaches to the right",
    "points forward",
]

STYLES = [
    "slowly",
    "quickly",
    "carefully",
    "energetically",
    "smoothly",
    "with small steps",
    "with wide steps",
    "with a heavy motion",
    "with a relaxed motion",
    "with a playful motion",
    "while staying balanced",
    "while keeping the torso upright",
]

DIRECTIONS = [
    "toward the front",
    "toward the back",
    "to the left",
    "to the right",
    "diagonally forward",
    "diagonally backward",
    "then returns to the start",
    "then faces forward",
]

TRANSITIONS = [
    ("walks forward", "turns left"),
    ("walks forward", "turns right"),
    ("walks backward", "turns around"),
    ("steps sideways", "waves"),
    ("jumps", "lands softly"),
    ("squats down", "stands up"),
    ("turns around", "waves"),
    ("kicks with one leg", "steps back"),
    ("raises both arms", "lowers them"),
    ("leans forward", "recovers"),
    ("leans backward", "recovers"),
    ("dances happily", "spins once"),
    ("jogs in place", "slows down"),
    ("walks in a circle", "stops"),
]


def build_prompts(num_prompts: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []

    for action in BASE_ACTIONS:
        prompts.append(f"A person {action}.")
        for style in STYLES:
            prompts.append(f"A person {action} {style}.")
        for direction in DIRECTIONS:
            prompts.append(f"A person {action} {direction}.")

    for first, second in TRANSITIONS:
        prompts.append(f"A person {first} and then {second}.")
        for style in STYLES:
            prompts.append(f"A person {first} {style} and then {second}.")
        for direction in DIRECTIONS:
            prompts.append(f"A person {first} {direction} and then {second}.")

    composed_actions = list(BASE_ACTIONS)
    rng.shuffle(composed_actions)
    while len(prompts) < num_prompts * 2:
        first = rng.choice(BASE_ACTIONS)
        second = rng.choice(BASE_ACTIONS)
        style = rng.choice(STYLES)
        direction = rng.choice(DIRECTIONS)
        template = rng.choice(
            [
                "A person {first} {style}, then {second}.",
                "A person {first}, {second}, and faces forward.",
                "A person {first} {direction}, then {second} {style}.",
                "A person starts by {first} and finishes by {second}.",
            ]
        )
        prompts.append(template.format(first=first, second=second, style=style, direction=direction))

    unique = list(dict.fromkeys(prompts))
    rng.shuffle(unique)
    return unique[:num_prompts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1")

    prompts = build_prompts(args.num_prompts, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated ARDY-Mamba text-to-motion distillation prompts.",
        f"# num_prompts={len(prompts)} seed={args.seed}",
        *prompts,
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(prompts)} prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
