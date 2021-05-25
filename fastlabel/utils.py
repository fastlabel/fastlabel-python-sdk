import base64


def base64_encode(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def is_image_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))


def is_video_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(('.mp4'))
