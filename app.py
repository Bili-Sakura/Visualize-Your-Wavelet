from __future__ import annotations

from typing import Callable, Dict, List

import gradio as gr
import numpy as np
from PIL import Image

WaveletFn = Callable[[np.ndarray], Dict[str, np.ndarray]]

COMPONENT_ORDER: List[str] = ["LL", "LH", "HL", "HH"]


def _ensure_even(image: Image.Image) -> Image.Image:
    width, height = image.size
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    if (even_width, even_height) != (width, height):
        image = image.crop((0, 0, even_width, even_height))
    return image


def _prepare_grayscale(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L")
    grayscale = _ensure_even(grayscale)
    width, height = grayscale.size
    if width < 2 or height < 2:
        raise gr.Error("Image must be at least 2x2 pixels after cropping.")
    return np.asarray(grayscale, dtype=np.float32)


def _normalize_component(component: np.ndarray) -> np.ndarray:
    min_value = float(component.min())
    max_value = float(component.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(component, dtype=np.uint8)
    normalized = (component - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def haar_wavelet_components(image_array: np.ndarray) -> Dict[str, np.ndarray]:
    a = image_array[0::2, 0::2]
    b = image_array[0::2, 1::2]
    c = image_array[1::2, 0::2]
    d = image_array[1::2, 1::2]

    ll = (a + b + c + d) / 2.0
    lh = (-a - b + c + d) / 2.0
    hl = (-a + b - c + d) / 2.0
    hh = (a - b - c + d) / 2.0

    return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


WAVELET_METHODS: Dict[str, WaveletFn] = {"Haar": haar_wavelet_components}


def compute_wavelet(
    image: Image.Image | None, method_name: str
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, Image.Image | None]:
    if image is None:
        return (None, None, None, None)
    method = WAVELET_METHODS.get(method_name)
    if method is None:
        raise gr.Error(f"Unknown wavelet method: {method_name}")

    grayscale = _prepare_grayscale(image)
    components = method(grayscale)

    outputs: List[Image.Image] = []
    for key in COMPONENT_ORDER:
        component = components[key]
        normalized = _normalize_component(component)
        outputs.append(Image.fromarray(normalized, mode="L"))
    return tuple(outputs)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Visualize Your Wavelet") as demo:
        gr.Markdown("## Visualize Your Wavelet")
        gr.Markdown(
            "Upload an image to view its Haar wavelet components. "
            "Images are cropped to even dimensions for the transform."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image")
            method = gr.Dropdown(
                choices=list(WAVELET_METHODS.keys()),
                value="Haar",
                label="Wavelet Method",
            )
        run_button = gr.Button("Compute Wavelet")
        with gr.Row():
            ll_image = gr.Image(
                label="LL (Approximation)", show_download_button=True
            )
            lh_image = gr.Image(
                label="LH (Vertical Details)", show_download_button=True
            )
        with gr.Row():
            hl_image = gr.Image(
                label="HL (Horizontal Details)", show_download_button=True
            )
            hh_image = gr.Image(
                label="HH (Diagonal Details)", show_download_button=True
            )

        run_button.click(
            fn=compute_wavelet,
            inputs=[input_image, method],
            outputs=[ll_image, lh_image, hl_image, hh_image],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
