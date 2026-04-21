#!/usr/bin/env python3
"""
generate_negatives.py - Generate "not a fruit" training images.

Creates 2000 synthetic non-fruit images using:
- Random noise/gradients
- Solid colors / patterns
- Random texture patches
These serve as the 'other' class for Phase 0 fruit type detection.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_fruits', 'other')
NUM_IMAGES = 2000
SIZE = 128


def make_noise(rng):
    """Uniform random noise."""
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def make_gradient(rng):
    """Smooth linear gradient in random direction."""
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    c2 = rng.integers(0, 256, 3).astype(np.float32)
    if rng.random() > 0.5:
        t = np.linspace(0, 1, SIZE).reshape(1, -1, 1)
    else:
        t = np.linspace(0, 1, SIZE).reshape(-1, 1, 1)
    img = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    return np.broadcast_to(img, (SIZE, SIZE, 3)).copy()


def make_solid(rng):
    """Solid colour with slight noise."""
    color = rng.integers(0, 256, 3, dtype=np.uint8)
    img = np.full((SIZE, SIZE, 3), color, dtype=np.uint8)
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_stripes(rng):
    """Random stripes pattern."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    c1 = rng.integers(0, 256, 3, dtype=np.uint8)
    c2 = rng.integers(0, 256, 3, dtype=np.uint8)
    stripe_w = rng.integers(4, 20)
    for x in range(0, SIZE, stripe_w):
        color = c1 if (x // stripe_w) % 2 == 0 else c2
        img[:, x:x+stripe_w] = color
    return img


def make_circles(rng):
    """Random circles on a background."""
    bg = rng.integers(0, 256, 3, dtype=np.uint8)
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    for _ in range(rng.integers(3, 12)):
        cx, cy = rng.integers(0, SIZE, 2)
        r = rng.integers(5, 40)
        color = tuple(int(x) for x in rng.integers(0, 256, 3))
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
    return img


def make_text(rng):
    """Random text on background."""
    bg = rng.integers(180, 256, 3, dtype=np.uint8)
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    words = ['hello', 'test', 'abc', '123', 'data', 'none', 'xyz', 'foo']
    for _ in range(rng.integers(2, 6)):
        word = words[rng.integers(0, len(words))]
        x, y = rng.integers(5, SIZE-30, 2)
        color = tuple(int(x) for x in rng.integers(0, 100, 3))
        cv2.putText(img, word, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + rng.random() * 0.4, color, 1)
    return img


def make_checkerboard(rng):
    """Checkerboard pattern."""
    c1 = rng.integers(0, 256, 3, dtype=np.uint8)
    c2 = rng.integers(0, 256, 3, dtype=np.uint8)
    sq = rng.integers(8, 24)
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(0, SIZE, sq):
        for x in range(0, SIZE, sq):
            color = c1 if ((y // sq) + (x // sq)) % 2 == 0 else c2
            img[y:y+sq, x:x+sq] = color
    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    generators = [make_noise, make_gradient, make_solid, make_stripes,
                  make_circles, make_text, make_checkerboard]

    print(f"Generating {NUM_IMAGES} non-fruit images...")
    for i in range(NUM_IMAGES):
        gen = generators[rng.integers(0, len(generators))]
        img = gen(rng)
        path = os.path.join(OUTPUT_DIR, f"neg_{i:05d}.jpg")
        cv2.imwrite(path, img)

    count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    print(f"Done! {count} images saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
