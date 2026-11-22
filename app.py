"""API propia de lectura de placas (Tesseract + OpenCV) para ParkGate AI.

POST /read   (multipart, campo "image")  ->  {"plate": "KTM-437" | null, "candidates": [...], "ms": 412}
GET  /       ->  {"ok": true}
Si defines la variable API_TOKEN, se exige la cabecera  Authorization: Bearer <API_TOKEN>.
"""
import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # Tesseract con 1 hilo: evita ahogarse en CPU limitada
import re
import time

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

app = FastAPI(title="ParkGate OCR")
API_TOKEN = os.getenv("API_TOKEN", "")
BUDGET_S = float(os.getenv("BUDGET_S", "14"))   # tiempo máximo de lectura (s)
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
HYPHEN = re.compile(r"([A-Z0-9]{3})\s*[-–—]\s*([A-Z0-9]{3})")
PLAIN = re.compile(r"([A-Z0-9]{3})\s*([A-Z0-9]{3})")


def parse(text: str):
    t = re.sub(r"PER[UÚ]", " ", text.upper())
    m = HYPHEN.search(t) or PLAIN.search(t)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def plate_crops(img):
    """Recortes candidatos: primero las zonas claras con proporción de placa y al final la imagen completa."""
    h, w = img.shape[:2]
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
    crops = []
    for _, x, y, bw, bh in sorted(boxes, reverse=True)[:2]:
        pad = int(0.04 * max(bw, bh))
        crops.append(img[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad])
    crops.append(img)
    return crops


def variants(crop):
    h, w = crop.shape[:2]
    # ~700 px de lado mayor: suficiente para Tesseract y mucho más rápido en CPU limitada
    scale = 700 / max(h, w)
    if abs(scale - 1) > 0.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield gray
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield cv2.createCLAHE(2.0, (8, 8)).apply(gray)


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
    runs = 0
    done = False
    for crop in plate_crops(img):
        for v in variants(crop):
            for psm in (6, 7):
                if time.perf_counter() - started > BUDGET_S:
                    done = True
                    break
                txt = pytesseract.image_to_string(v, config=f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}")
                runs += 1
                p = parse(txt)
                if p:
                    votes[p] = votes.get(p, 0) + 1
                    if votes[p] >= 2:      # dos lecturas iguales: suficiente
                        done = True
                        break
            if done:
                break
        if done:
            break
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    return {
        "plate": ranked[0][0] if ranked else None,
        "candidates": [{"plate": p, "votes": n} for p, n in ranked[:4]],
        "runs": runs,
        "ms": round((time.perf_counter() - started) * 1000),
    }
