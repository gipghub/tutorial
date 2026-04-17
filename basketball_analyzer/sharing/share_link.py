from __future__ import annotations

from pathlib import Path


def generate_share_url(host: str, job_id: str, port: int = 5000) -> str:
    return f"http://{host}:{port}/share/{job_id}"


def generate_qr(url: str, output_path: Path) -> Path:
    import qrcode  # type: ignore
    from PIL import Image

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img: Image.Image = qr.make_image(fill_color="black", back_color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    return output_path
