import os
import re
import hashlib
from urllib.parse import urlparse, unquote, parse_qs
from PIL import Image
import numpy as np
import requests


def url2path(url, img_cache_dir):
    try:
        # Generate a hash of the URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        parsed_url = urlparse(url)
        base_filename = os.path.basename(parsed_url.path)
        base_filename = unquote(base_filename)
        base_filename = base_filename.split('?')[0]
        base_filename = re.sub(r'[^A-Za-z0-9._-]', '_', base_filename)
        ext = os.path.splitext(base_filename)[1].lower()
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            query_params = parse_qs(parsed_url.query)
            if 'format' in query_params:
                format_param = query_params['format'][0].lower()
                if format_param in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    ext = f'.{format_param}'
            else:
                ext = '.jpg'
        filename_length = len(base_filename)
        if not base_filename or base_filename == 'image' or filename_length > 200:
            filename = f'image_{url_hash}{ext}'
        else:
            base_filename = os.path.splitext(base_filename)[0]
            filename = f'{base_filename}_{url_hash}{ext}'
        local_path = img_cache_dir / filename
        if local_path.is_dir():
            local_path = local_path.with_name(f'{local_path.name}_file{ext}')
        return local_path
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return img_cache_dir / f'image_{hashlib.md5(url.encode()).hexdigest()}.jpg'

def download_and_cache_image(example, img_cache_dir):
    try:
        image_url = example["image_url"]
        local_path = url2path(image_url, img_cache_dir)
        if not local_path.exists():
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return {"local_image_path": str(local_path), "download_status": "ok"}
    except Exception as e:
        return {"local_image_path": None, "download_status": "error"}


def check_image(x):
    try:
        img = Image.open(x["local_image_path"]).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3 or img.shape[1] == 1:
            return False
        return True
    except:
        return False
    

def points_to_text(points, label_text, alt_text):
    if len(points) == 1:
        x_str, y_str = points[0]
        return f"<point x=\"{x_str}\" y=\"{y_str}\" alt=\"{alt_text}\">{label_text}</point>"
    point_text = []
    for ix, (x, y) in enumerate(points, start=1):
        point_text.append(f"x{ix}=\"{x}\"")
        point_text.append(f"y{ix}=\"{y}\"")
    point_text = " ".join(point_text)
    return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"

def text_to_points(text):
    single_point_pattern = r'^<point\s+x="([^"]+)"\s+y="([^"]+)"\s+alt="([^"]+)">(.*?)</point>$'
    m = re.match(single_point_pattern, text, re.DOTALL)
    if m:
        x_str, y_str, alt_text, label_text = m.groups()
        points = [(x_str, y_str)]
        return points, label_text, alt_text
    if text.startswith("<points"):
        alt_match = re.search(r'alt="([^"]+)"', text)
        label_match = re.search(r'>(.*?)</points>', text, re.DOTALL)
        if not alt_match or not label_match: raise ValueError("Invalid format for multiple points")
        alt_text = alt_match.group(1)
        label_text = label_match.group(1)
        x_matches = re.findall(r'x(\d+)="([^"]+)"', text)
        y_matches = re.findall(r'y(\d+)="([^"]+)"', text)
        x_dict = {int(idx): val for idx, val in x_matches}
        y_dict = {int(idx): val for idx, val in y_matches}
        points = []
        for i in sorted(x_dict.keys()):
            if i in y_dict: points.append((x_dict[i], y_dict[i]))
            else: raise ValueError(f"Missing y{i} value")
        return points, label_text, alt_text
    raise ValueError("Input string does not start with <point> or <points>")
