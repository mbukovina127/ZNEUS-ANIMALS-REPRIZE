import base64
import os

import numpy as np
from IPython.display import HTML, display

#### ---------------------------------------------
#### Funkcie tykajuce sa obrazkov
#### ---------------------------------------------
def sample_per_class(df, n=3):

    groups = df.groupby("label")
    sampled = groups.apply(lambda g: g.sample(min(n, len(g)), random_state=42))
    sampled = sampled.reset_index(drop=True)
    return sampled

def img_to_html(path, width=220):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"<img src='data:image/jpeg;base64,{data}' style='width:{width}px; height:auto;'/>"

def show_images(df, title="Výber obrázkov", n=None, width=220):

    if n is not None:
        df = df.head(n)

    html = f"<h2>{title}</h2><div style='display:flex; flex-wrap:wrap;'>"

    for row in df.itertuples():

        folder = os.path.basename(os.path.dirname(row.filepath))
        filename = os.path.basename(row.filepath)
        short_path = f"{folder}/{filename}"

        html += f"""
        <div style='margin:10px; width:{width}px; text-align:center;'>

            {img_to_html(row.filepath, width)}

            <div style='
                font-size:12px;
                margin-top:4px;
                max-width:{width}px;
                word-wrap:break-word;
            '>
                <b>{row.label}</b><br>
                {short_path}<br>
                {row.width}×{row.height}px<br>
                brightness={row.brightness:.1f}<br>
                {row.format}, {row.color_mode}, {row.bit_depth}-bit
            </div>

        </div>
        """

    html += "</div>"
    display(HTML(html))

def map_mode_to_depth(row):
    mode = row["color_mode"]
    depth = row["bit_depth"]

    if mode == "RGB":
        return "8-bit RGB"
    if mode == "RGBA":
        return "8-bit RGBA"
    if mode == "L":
        return "8-bit Grayscale"
    if mode == "I;16":
        return "16-bit Grayscale"

    return f"other ({mode})"