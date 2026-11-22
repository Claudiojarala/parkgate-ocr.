"""API propia de lectura de placas (Tesseract + OpenCV) para ParkGate AI.

POST /read   (multipart, campo "image")  ->  {"plate": "KTM-437" | null, "candidates": [...], "ms": 412}
GET  /       ->  {"ok": true}
Si defines la variable API_TOKEN, se exige la cabecera  Authorization: Bearer <API_TOKEN>.
"""
import os
import re
import time

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

app = FastAPI(title="ParkGate OCR")
API_TOKEN = os.getenv("API_TOKEN", "")
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
HYPHEN = re.compile(r"([A-Z0-9]{3})\s*[-–—]\s*([A-Z0-9]{3})")
PLAIN = re.compile(r"([A-Z0-9]{3})\s*([A-Z0-9]{3})")


def parse(text: str):
    t = re.sub(r"PER[UÚ]", " ", text.upper())
    m = HYPHEN.search(t) or PLAIN.search(t)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def plate_crops(img):
    """Recortes candidatos: la imagen completa y las zonas claras con proporción de placa."""
    h, w = img.shape[:2]
    crops = [img]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.02 * w * h or area > 0.95 * w * h or bh == 0:
            continue
        if 1.3 <= bw / bh <= 3.6:
            boxes.append((area, x, y, bw, bh))
    for _, x, y, bw, bh in sorted(boxes, reverse=True)[:3]:
        pad = int(0.04 * max(bw, bh))
        crops.append(img[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad])
    return crops


def variants(crop):
    h, w = crop.shape[:2]
    scale = 900 / max(h, w) if max(h, w) < 900 else min(1.0, 1600 / max(h, w))
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    out = [gray, cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    out.append(cv2.createCLAHE(2.0, (8, 8)).apply(gray))
    return out


@app.get("/")
def health():
    return {"ok": True, "service": "parkgate-ocr"}


@app.post("/read")
async def read(image: UploadFile = File(...), authorization: str = Header(default="")):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="token inválido")
    started = time.perf_counter()
    data = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="imagen no válida")

    votes: dict[str, int] = {}
    for crop in plate_crops(img):
        for v in variants(crop):
            for psm in (7, 6):
                txt = pytesseract.image_to_string(v, config=f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}")
                p = parse(txt)
                if p:
                    votes[p] = votes.get(p, 0) + 1
        if votes and max(votes.values()) >= 3:
            break
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    return {
        "plate": ranked[0][0] if ranked else None,
        "candidates": [{"plate": p, "votes": n} for p, n in ranked[:4]],
        "ms": round((time.perf_counter() - started) * 1000),
    }
