"""Fase 0: configuracion, lectura del video, telemetria, deteccion del evento, modelo de
fondo, camara georreferenciada y procedencia de la corrida.
"""

import gc
import hashlib
import json
import math
import os
import re
import shutil
import struct
import pickle
import time
from dataclasses import dataclass, field, asdict, fields

import cv2
import numpy as np


def raiz_por_defecto() -> str:
    """Carpeta de trabajo actual = carpeta del notebook."""
    return os.getcwd()


FASES = {
    0: "0_preprocesamiento",
    1: "1_deteccion_gas",
    2: "2_reconstruccion3d",
    3: "3_viento",
    4: "4_pmx",
}

@dataclass
class Config:
    raiz: str = field(default_factory=raiz_por_defecto)
    run: str = ""

    video_nombre: str = ""
    dem_nombre: str = "DEM_mina.tif"
    satelital_nombre: str = "satelital_utm.tif"
    calibracion_nombre: str = "auto"

    sam2_ckpt_nombre: str = "sam2_hiera_tiny.pt"
    sam2_cfg: str = "configs/sam2/sam2_hiera_t.yaml"
    clipseg_id: str = "CIDAS/clipseg-rd64-refined"
    dispositivo: str = "cpu"

    escala: float = 0.5
    pivot_frame: int | None = None
    analisis_fin: int | None = None
    gsd_m_px: float = None

    frame_data: dict | None = None
    focal_px_por_ancho: dict = field(default_factory=lambda: {3840: 2700,
                                                             1920: 1350})
    ts_inicio_video: str | None = None
    tz_local: str = "America/Santiago"
    bounds_mina: tuple = (506500.0, 7532500.0, 513500.0, 7538500.0)

    usar_cache: bool = True

    def _dato(self, nombre: str) -> str:
        return nombre if os.path.isabs(nombre) else os.path.join(self.dir_datos, nombre)

    @property
    def dir_datos(self) -> str:
        return os.path.join(self.raiz, "datos")

    @property
    def dir_resultados(self) -> str:
        """resultados/  o  resultados/<run>/ si `run` no esta vacio."""
        base = os.path.join(self.raiz, "resultados")
        return os.path.join(base, self.run) if self.run else base

    @property
    def dir_cache(self) -> str:
        return os.path.join(self.dir_resultados, "cache")

    @property
    def video(self) -> str:
        return self._dato(self.video_nombre)

    @property
    def sam2_ckpt(self) -> str:
        return self._dato(self.sam2_ckpt_nombre)

    @property
    def dir_clipseg(self) -> str:
        return self._dato(self.clipseg_id.split("/")[-1])

    @property
    def dem(self) -> str:
        return self._dato(self.dem_nombre)

    @property
    def satelital(self) -> str:
        return self._dato(self.satelital_nombre)

    @property
    def calibracion(self) -> str:
        """Ruta del .pkl de la calibracion del airblast."""
        nombre = self.calibracion_nombre
        if not nombre or nombre == "auto":
            nombre = nombre_calibracion(self.video_nombre)
        return nombre if os.path.isabs(nombre) else os.path.join(self.dir_fase(0), nombre)

    def dir_fase(self, n: int, *sub: str) -> str:
        """Carpeta de salida de una fase. Crea los subdirectorios pedidos."""
        if n not in FASES:
            raise KeyError(f"No existe la fase {n}. Las fases son {sorted(FASES)}.")
        d = os.path.join(self.dir_resultados, FASES[n], *sub)
        os.makedirs(d, exist_ok=True)
        return d

    def ruta(self, n: int, *partes: str) -> str:
        """Ruta de un archivo dentro de la carpeta de una fase."""
        return os.path.join(self.dir_fase(n, *partes[:-1]), partes[-1])

    @property
    def art_fondo(self) -> str:
        return self.ruta(0, "modelo_fondo.npz")

    @property
    def art_video_meta(self) -> str:
        return self.ruta(0, "video_meta.json")

    @property
    def art_onset(self) -> str:
        """Onset de la tronadura y ventana tranquila (escaneo de actividad)."""
        return self.ruta(0, "onset.json")

    @property
    def art_geometria(self) -> str:
        """Pivote en px y GSD, resueltos desde la calibracion del airblast."""
        return self.ruta(0, "geometria.json")

    @property
    def art_georreferencia(self) -> str:
        """Camara en UTM y verificaciones de la georreferencia."""
        return self.ruta(0, "georreferencia.json")

    @property
    def art_mascaras_gas(self) -> str:
        return self.ruta(1, "mascaras_gas.pkl")

    @property
    def art_mascaras_sombra(self) -> str:
        return self.ruta(1, "mascaras_sombra.pkl")

    @property
    def art_viento(self) -> str:
        return self.ruta(3, "viento_flujo_optico.csv")

    @property
    def art_serie_columna(self) -> str:
        return self.ruta(2, "serie_columna.csv")

    @property
    def art_gas_glb(self) -> str:
        return self.ruta(2, "gas_secuencia.glb")

    def crear_directorios(self) -> "Config":
        os.makedirs(self.dir_datos, exist_ok=True)
        os.makedirs(self.dir_cache, exist_ok=True)
        for n in FASES:
            self.dir_fase(n)
        return self

    def guardar_json(self, ruta: str | None = None) -> str:
        ruta = ruta or os.path.join(self.dir_resultados, "config.json")
        d = asdict(self)
        d["_rutas"] = {"video": self.video, "dem": self.dem,
                       "resultados": self.dir_resultados}
        with open(ruta, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, ensure_ascii=False)
        return ruta

    @classmethod
    def desde_json(cls, ruta: str) -> "Config":
        """Reconstruye la Config guardada por la Fase 0."""
        with open(ruta, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        validos = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in validos}
        if "bounds_mina" in d:
            d["bounds_mina"] = tuple(d["bounds_mina"])
        return cls(**d)

    def resumen(self) -> None:
        print(f"raiz        : {ruta_corta(self.raiz)}")
        print(f"run         : {self.run or '(sin subcarpeta de version)'}")
        print(f"resultados  : {ruta_corta(self.dir_resultados)}")
        print(f"video       : {ruta_corta(self.video)}"
              f"{'' if os.path.exists(self.video) else '   [NO EXISTE]'}")
        piv = (f"f{self.pivot_frame}" if self.pivot_frame is not None
               else "(sin resolver: lo pone C0b/C1b)")
        print(f"escala      : {self.escala}  |  pivote {piv}"
              f"  |  GSD {self.gsd_m_px} m/px")


def iniciar(run: str = "", crear: bool = False, raiz: str | None = None,
            **overrides) -> dict:
    """Punto de entrada unico de TODOS los notebooks."""
    raiz = raiz or raiz_por_defecto()
    _base = os.path.join(raiz, "resultados")
    ruta_cfg = os.path.join(_base, run, "config.json") if run \
        else os.path.join(_base, "config.json")

    if crear:
        cfg = Config(raiz=raiz, run=run, **overrides)
        if not cfg.video_nombre:
            raise ValueError(
                "Falta el nombre del video. Fijalo en la celda VIDEO_NOMBRE del "
                "notebook y pasalo a F0.iniciar(..., video_nombre=VIDEO_NOMBRE).")
        cfg.crear_directorios()
        if not os.path.exists(cfg.video):
            raise FileNotFoundError(
                f"No encuentro el video en:\n    {cfg.video}\n\n"
                f"La raiz del proyecto es la carpeta de trabajo del notebook:\n"
                f"    {cfg.raiz}\n"
                f"Deja el video en {os.path.join(cfg.raiz, 'datos')}, o pasa una "
                f"ruta absoluta:\n"
                f"    F0.iniciar(..., video_nombre=r'D:\\ruta\\al\\video.mp4')\n"
                f"o cambia la raiz:  F0.iniciar(..., raiz=r'C:\\...\\FASES')")
        invalidar_si_cambio_video(cfg)
        proc = resolver_evento(cfg)
        cfg.guardar_json()
        sellar(cfg, 0, dict(procedencia=proc))
        print(f"\n[config] creada y guardada en {ruta_corta(ruta_cfg)}\n")
    else:
        if not os.path.exists(ruta_cfg):
            raise FileNotFoundError(
                f"No encuentro {ruta_cfg}.\n"
                f"Corre primero el notebook de la Fase 0 con "
                f"F0.iniciar(run={run!r}, crear=True).")
        cfg = Config.desde_json(ruta_cfg)
        if not os.path.isdir(cfg.raiz):
            print(f"[config] la raiz guardada ({ruta_corta(cfg.raiz)}) ya no existe; "
                  f"uso {ruta_corta(raiz)}")
            cfg.raiz = raiz
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.crear_directorios()
        print(f"[config] leida de {ruta_corta(ruta_cfg)}\n")
        _sello = _leer_procedencia(cfg).get("video")
        if _sello and _sello != os.path.basename(cfg.video_nombre):
            print(f"\nLa config guardada apunta a '{_sello}'. Si cambiaste de "
                  f"video, corre la Fase 0 con crear=True.")

    cfg.resumen()
    if not os.path.exists(cfg.video):
        raise FileNotFoundError(
            f"No encuentro el video en:\n    {cfg.video}\n\n"
            f"La raiz del proyecto es la carpeta de trabajo del notebook:\n"
            f"    {cfg.raiz}\n"
            f"Deja el video en {os.path.join(cfg.raiz, 'datos')}, o pasa una "
            f"ruta absoluta:\n"
            f"    F0.iniciar(..., video_nombre=r'D:\\ruta\\al\\video.mp4')\n"
            f"o cambia la raiz:  F0.iniciar(..., raiz=r'C:\\...\\FASES')")
    vid = propiedades_video(cfg.video)
    fin = cfg.analisis_fin or vid["n_frames"]
    h = int(vid["alto"] * cfg.escala)
    w = int(vid["ancho"] * cfg.escala)
    print(f"video   : {vid['ancho']}x{vid['alto']} | {vid['fps']:.2f} fps | "
          f"{vid['n_frames']} frames")
    _piv = "?" if cfg.pivot_frame is None else cfg.pivot_frame
    print(f"trabajo : {w}x{h} | analisis [{_piv}, {fin})")
    return dict(CFG=cfg, VID=vid, FPS=vid["fps"], N_FRAMES=vid["n_frames"],
                FIN=fin, H_WORK=h, W_WORK=w, FORMA=(h, w))


def ruta_corta(p) -> str:
    """Ruta relativa a la carpeta de trabajo; si queda fuera de ella, solo el nombre."""
    p = os.path.abspath(str(p))
    try:
        r = os.path.relpath(p, os.getcwd())
    except ValueError:
        return os.path.basename(p)
    return os.path.basename(p) if r.startswith("..") else r


def hash_parametros(params: dict, n: int = 8) -> str:
    """MD5 corto de un dict de parametros."""
    txt = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:n]


def guardar_cache(obj, nombre: str, dir_cache: str, verbose: bool = True) -> str:
    os.makedirs(dir_cache, exist_ok=True)
    ruta = os.path.join(dir_cache, f"{nombre}.pkl")
    with open(ruta, "wb") as fh:
        pickle.dump(obj, fh)
    if verbose:
        print(f"[cache] guardado: {os.path.basename(ruta)} "
              f"({os.path.getsize(ruta)/1e6:.1f} MB)")
    return ruta


def cargar_cache(nombre: str, dir_cache: str, verbose: bool = True):
    ruta = os.path.join(dir_cache, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        return None
    with open(ruta, "rb") as fh:
        obj = pickle.load(fh)
    if verbose:
        print(f"[cache] cargado: {os.path.basename(ruta)}")
    return obj


def propiedades_video(ruta: str) -> dict:
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")
    d = dict(fps=float(cap.get(cv2.CAP_PROP_FPS)),
             n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
             ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return d


class LectorVideo:
    """Lector con seek que mantiene el `VideoCapture` abierto."""

    def __init__(self, ruta: str, escala: float = 1.0):
        self.ruta, self.escala = ruta, escala
        self.cap = cv2.VideoCapture(ruta)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _escalar(self, frame):
        if self.escala == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self.escala, fy=self.escala)

    def __getitem__(self, fi: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = self.cap.read()
        return self._escalar(frame) if ok else None

    def recorrer(self, frames):
        """Itera (frame_idx, imagen) leyendo secuencialmente cuando se puede."""
        frames = sorted(frames)
        if not frames:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        objetivo, k = set(frames), 0
        while k < len(frames):
            fi = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if fi > frames[-1]:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            if fi in objetivo:
                k += 1
                yield fi, self._escalar(frame)

    def forma_trabajo(self) -> tuple:
        f0 = self[0]
        return f0.shape[:2]

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def escritor_video(ruta: str, fps: float, ancho: int, alto: int):
    """VideoWriter mp4v con dimensiones forzadas a par (requisito del codec)."""
    w, h = ancho & ~1, alto & ~1
    vw = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"No se pudo abrir el VideoWriter en {ruta}")
    return vw, (w, h)


def empaquetar(m) -> np.ndarray:
    """Mascara booleana -> bits. 1920x1080 uint8 = 2 MB; empaquetada = 260 KB."""
    return np.packbits(np.asarray(m, dtype=bool), axis=None)


def desempaquetar(b: np.ndarray, forma: tuple) -> np.ndarray:
    return (np.unpackbits(b, count=forma[0] * forma[1])
            .reshape(forma).astype(np.uint8) * 255)


def normalizar_mascara(m, forma: tuple | None = None) -> np.ndarray:
    """A uint8 {0,255}, opcionalmente reescalada a `forma` (nearest)."""
    m = np.asarray(m)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() == 1:
        m = m * 255
    if forma is not None and m.shape != forma:
        m = cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST)
    return m


def guardar_mascaras(masks: dict, ruta: str, meta: dict | None = None,
                     empaquetar_bits: bool = False) -> str:
    """Guarda {frame: mascara} en pkl con metadatos del algoritmo que las creo."""
    meta = dict(meta or {})
    if empaquetar_bits:
        forma = np.asarray(next(iter(masks.values()))).shape
        masks = {int(f): empaquetar(m) for f, m in masks.items()}
        meta.update(empaquetado=True, forma=tuple(int(x) for x in forma))
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as fh:
        pickle.dump(dict(masks=masks, meta=meta), fh)
    print(f"[mascaras] {len(masks)} frames -> {ruta_corta(ruta)} "
          f"({os.path.getsize(ruta)/1e6:.1f} MB"
          + (", empaquetadas a bits" if empaquetar_bits else "") + ")")
    return ruta


def cargar_mascaras(ruta: str) -> tuple:
    """Devuelve (masks, meta). Acepta pkl antiguos que sean un dict pelado."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta el artefacto de mascaras: {ruta}")
    with open(ruta, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "masks" in obj and "meta" in obj:
        masks, meta = obj["masks"], obj["meta"]
    else:
        return obj, {}
    if meta.get("empaquetado"):
        forma = tuple(meta["forma"])
        masks = {int(f): desempaquetar(b_, forma) for f, b_ in masks.items()}
    return masks, meta


def metricas_mascara(m, area_min: int = 500) -> dict | None:
    """Area, centroide, segundos momentos centrales y contacto con el borde."""
    m = np.asarray(m)
    ys, xs = np.where(m > 0)
    if len(xs) < area_min:
        return None
    H, W = m.shape
    cx, cy = xs.mean(), ys.mean()
    n_borde = int(((xs == 0) | (xs == W - 1) | (ys == 0) | (ys == H - 1)).sum())
    return dict(area=len(xs), cx=float(cx), cy=float(cy),
                sxx=float(((xs - cx) ** 2).mean()),
                syy=float(((ys - cy) ** 2).mean()),
                sxy=float(((xs - cx) * (ys - cy)).mean()),
                frac_borde=n_borde / (2.0 * (H + W)),
                cobertura=len(xs) / (H * W))


def guardar_csv(filas: list, columnas: list, ruta: str) -> str:
    import csv
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columnas)
        w.writerows(filas)
    print(f"[csv] {len(filas)} filas -> {ruta_corta(ruta)}")
    return ruta


_MARCAS_NATIVAS = ("_nat", "_native", "_nativo", "_4k")

ESQUEMA_GEOMETRIA = 2


def cargar_calibracion(ruta: str) -> dict:
    """Calibracion del airblast como dict. Acepta .pkl (pickle) y .json."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No encuentro la calibracion del airblast:\n    {ruta}\n"
            f"La produce la celda F0-3 y la guarda en resultados/0_preprocesamiento/ "
            f"con el nombre canonico del video. Si la tienes de una corrida vieja "
            f"con otro nombre, copiala ahi (F0 la renombra sola si "
            f"declara el video por dentro) o apunta "
            f"Config.calibracion_nombre a su ruta.")
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".json":
        with open(ruta, "r", encoding="utf-8") as fh:
            cal = json.load(fh)
    elif ext in (".pkl", ".pickle"):
        with open(ruta, "rb") as fh:
            cal = pickle.load(fh)
    else:
        with open(ruta, "rb") as fh:
            cabecera = fh.read(1)
        if cabecera == b"\x80":
            with open(ruta, "rb") as fh:
                cal = pickle.load(fh)
        else:
            with open(ruta, "r", encoding="utf-8") as fh:
                cal = json.load(fh)
    if not isinstance(cal, dict):
        raise TypeError(f"La calibracion {ruta} no es un dict sino {type(cal)}.")
    return cal


def _num(v) -> float:
    """float desde int, np.float32, array de un elemento o str numerico."""
    return float(np.asarray(v).ravel()[0])


def _clave(cal: dict, *nombres):
    """(clave_real, valor) de la primera de `nombres` presente, sin distinguir mayusculas;
    (None, None) si no hay ninguna.
    """
    low = {str(k).lower(): k for k in cal}
    for n in nombres:
        k = low.get(n.lower())
        if k is not None:
            return k, cal[k]
    return None, None


def resolver_geometria(cfg, w_work: int, h_work: int, calib=None, onset=None,
                       verbose: bool = True) -> dict:
    """Pivote, GSD y frame de la detonacion a partir de la calibracion."""
    cal = cargar_calibracion(cfg.calibracion) if calib is None else calib
    esc = float(cfg.escala)
    origen = {}

    k, v = _clave(cal, "PIVOT_PX_WORK", "PIVOTE_PX_TRABAJO", "PIVOT_PX_TRABAJO")
    if v is not None and np.size(v) == 2:
        px_w, py_w = _num(np.ravel(v)[0]), _num(np.ravel(v)[1])
        px_n, py_n = px_w / esc, py_w / esc
        origen["pivote"] = f"{k} (ya en px de trabajo)"
    else:
        kx, vx = _clave(cal, "PIVOT_X_NAT", "PIVOTE_X_NAT", "CENTRO_X_NAT",
                        "PIVOT_X", "PIVOTE_X", "CENTRO_X")
        ky, vy = _clave(cal, "PIVOT_Y_NAT", "PIVOTE_Y_NAT", "CENTRO_Y_NAT",
                        "PIVOT_Y", "PIVOTE_Y", "CENTRO_Y")
        if vx is None or vy is None:
            raise RuntimeError(
                "No reconozco el pivote dentro de la calibracion. Claves "
                f"disponibles:\n    {sorted(map(str, cal))}\n"
                "Pasa el pivote a mano:  PIVOTE_PX_MANUAL = (x, y)  en px de "
                "TRABAJO.")
        px, py = _num(vx), _num(vy)
        marcada = (any(m in kx.lower() for m in _MARCAS_NATIVAS) and
                   any(m in ky.lower() for m in _MARCAS_NATIVAS))
        if marcada:
            px_n, py_n = px, py
            origen["pivote"] = f"{kx}/{ky} (nativos por el sufijo de la clave)"
        elif px >= w_work or py >= h_work:
            px_n, py_n = px, py
            origen["pivote"] = (f"{kx}/{ky} (nativos: no caben en el cuadro de "
                                f"trabajo {w_work}x{h_work})")
        else:
            px_n, py_n = px / esc, py / esc
            origen["pivote"] = f"{kx}/{ky} (se asumen px de trabajo)"
        px_w, py_w = px_n * esc, py_n * esc

    k, v = _clave(cal, "PPM_NAT", "PX_POR_METRO_NAT", "PPM", "PX_POR_METRO")
    ppm_nat = None
    if v is not None and _num(v) > 0:
        ppm_nat = _num(v)
        gsd_nat = 1.0 / ppm_nat
        origen["gsd"] = f"{k} = {ppm_nat:.4f} px/m (nativos)"
    else:
        k, v = _clave(cal, "GSD_NAT", "GSD_NATIVO", "GSD_M_PER_PX", "GSD")
        if v is not None and _num(v) > 0:
            gsd_nat = _num(v)
            ppm_nat = 1.0 / gsd_nat
            origen["gsd"] = f"{k} = {gsd_nat:.5f} m/px (nativos)"
        else:
            if cfg.gsd_nativo_m_px is None:
                raise RuntimeError(
                    "La calibracion no trae escala (ni PPM_NAT ni GSD_NAT) y "
                    "Config.gsd_nativo_m_px esta sin resolver. La escala metrica "
                    "se MIDE con el airblast: corre la celda de calibracion "
                    "antes de esta. Si de verdad quieres forzarla, pon el valor "
                    "en px NATIVOS.")
            gsd_nat = float(cfg.gsd_nativo_m_px)
            origen["gsd"] = (f"CFG.gsd_nativo_m_px = {gsd_nat} "
                                f"(forzado a mano; la calibracion no trae escala)")
    gsd_work = gsd_nat / esc

    k, v = _clave(cal, "T0_SUBFRAME", "T0", "BLAST_FRAME", "FRAME_DETONACION")
    if v is not None:
        t0 = _num(v)
        origen["t0"] = f"{k} = {t0:.3f}"
    elif cfg.pivot_frame is not None:
        t0 = float(cfg.pivot_frame)
        origen["t0"] = "CFG.pivot_frame (la calibracion no trae t0)"
    else:
        raise RuntimeError(
            "Ni la calibracion trae T0/BLAST_FRAME ni Config.pivot_frame esta "
            "resuelto. Corre C0b (escaneo de actividad) antes de esta celda.")
    blast = int(round(t0))

    _, v_i = _clave(cal, "PISO_INI", "QUIET_INI", "VENTANA_TRANQUILA_INI")
    _, v_f = _clave(cal, "PISO_FIN", "QUIET_FIN", "VENTANA_TRANQUILA_FIN")
    if onset is None and os.path.exists(cfg.art_onset):
        with open(cfg.art_onset, "r", encoding="utf-8") as fh:
            onset = json.load(fh)
    if v_i is not None and v_f is not None and int(_num(v_f)) - int(_num(v_i)) >= 5:
        piso = [int(_num(v_i)), int(_num(v_f))]
        origen["piso"] = "PISO_INI/PISO_FIN de la calibracion"
    elif onset and onset.get("piso_fin", 0) - onset.get("piso_ini", 0) >= 5:
        piso = [int(onset["piso_ini"]), int(onset["piso_fin"])]
        origen["piso"] = f"escaneo de actividad ({os.path.basename(cfg.art_onset)})"
    else:
        piso = None
        origen["piso"] = ("ni la calibracion ni el escaneo traen ventana "
                          "tranquila; la Fase 1 usara los frames previos al pivote")

    geo = dict(
        esquema=ESQUEMA_GEOMETRIA,
        escala=esc, forma_trabajo=[int(h_work), int(w_work)],
        pivote_px_nat=[px_n, py_n], pivote_px_trabajo=[px_w, py_w],
        ppm_nat=ppm_nat, gsd_nativo=gsd_nat, gsd_trabajo=gsd_work,
        t0_subframe=t0, blast_frame=blast, ventana_tranquila=piso,
        pivot_frame_config=(int(cfg.pivot_frame)
                            if cfg.pivot_frame is not None else blast),
        calibracion=os.path.basename(cfg.calibracion), origen=origen)

    if verbose:
        print(f"[geo] pivote  : nat ({px_n:.0f}, {py_n:.0f})  ->  trabajo "
              f"({px_w:.0f}, {py_w:.0f})   [{origen['pivote']}]")
        print(f"[geo] gsd     : {gsd_nat:.5f} m/px nativo  ->  {gsd_work:.5f} "
              f"m/px trabajo   [{origen['gsd']}]")
        print(f"[geo] cuadro  : {w_work * gsd_work:.0f} x {h_work * gsd_work:.0f} m"
              f"   <- si esto no es plausible, la escala esta mal")
        print(f"[geo] disparo : t0={t0:.3f} -> frame {blast}   "
              f"(CFG.pivot_frame={cfg.pivot_frame})")
        print(f"[geo] tranquila: {f'[f{piso[0]}, f{piso[1]})' if piso else '(no hay)'}"
              f"   [{origen['piso']}]")
        if cfg.pivot_frame is not None and abs(blast - cfg.pivot_frame) > 1:
            print(f"   AVISO: la calibracion pone la detonacion en f{blast} y "
                  f"Config en f{cfg.pivot_frame}. La ventana de busqueda del "
                  f"ancla y el fondo pre-tronadura se derivan de pivot_frame: "
                  f"si el desfase es real, corrige Config.")
        if not (0 <= px_w < w_work and 0 <= py_w < h_work):
            print(f"   AVISO: el pivote de trabajo cae FUERA del cuadro "
                  f"{w_work}x{h_work}. Revisa la escala o pasalo a mano.")
    return geo


PAR_ONSET = dict(
    escala=0.25,
    grid=(24, 16),
    ratio_k=2.5,
    ratio_min=3.0,
    k_intensidad=4.0,
    ep_hueco=5,
    ep_min=3,
    ep_fuerte=2.0,
    win=30,
    margen=10,
    tol=1.35,
)


def escaneo_actividad(cfg, par=None, forzar=False, verbose=True) -> dict:
    """Recorre el video entero midiendo actividad por bloques."""
    p = dict(PAR_ONSET)
    p.update(par or {})
    firma = hash_parametros(dict(video=os.path.basename(cfg.video),
                                 escala=p["escala"], grid=list(p["grid"])))
    nombre = f"escaneo_actividad_{firma}"
    if not forzar:
        sc = cargar_cache(nombre, cfg.dir_cache, verbose=verbose)
        if sc is not None:
            return sc

    gx, gy = int(p["grid"][0]), int(p["grid"][1])
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {cfg.video}")
    prev, act_l, bg_l = None, [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, None, fx=p["escala"], fy=p["escala"]),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            blk = cv2.resize(d, (gx, gy), interpolation=cv2.INTER_AREA)
            act_l.append(float(blk.max()))
            bg_l.append(float(np.median(blk)))
        prev = g
        if verbose and len(act_l) and len(act_l) % 500 == 0:
            print(f"   escaneo: {len(act_l)} frames")
    cap.release()
    if len(act_l) < 10:
        raise RuntimeError(f"Solo {len(act_l)} frames leidos de {cfg.video}.")
    sc = dict(act=np.asarray(act_l), bg=np.asarray(bg_l))
    guardar_cache(sc, nombre, cfg.dir_cache, verbose=verbose)
    return sc


def detectar_onset(sc: dict, fps: float, par=None, manual=None,
                   verbose=True) -> dict:
    """Onset de la tronadura y ventana tranquila, a partir del escaneo."""
    p = dict(PAR_ONSET)
    p.update(par or {})
    act, bg = np.asarray(sc["act"]), np.asarray(sc["bg"])
    fr_idx = np.arange(1, len(act) + 1)
    ratio = act / np.maximum(bg, 1e-3)

    rbase = float(np.median(ratio))
    th_rat = max(p["ratio_k"] * rbase, p["ratio_min"])
    loc = ratio > th_rat

    pool = act[~loc] if (~loc).sum() > 30 else act
    base = float(np.median(pool))
    mad = float(np.median(np.abs(pool - base))) * 1.4826
    th_act = base + p["k_intensidad"] * max(mad, 1e-6)
    cand = loc & (act > th_act)

    idx = np.where(cand)[0]
    episodios = []
    if len(idx):
        ini, prev = idx[0], idx[0]
        for k in idx[1:]:
            if k - prev > p["ep_hueco"]:
                episodios.append((ini, prev))
                ini = k
            prev = k
        episodios.append((ini, prev))
        episodios = [(a, b) for a, b in episodios
                     if (b - a + 1) >= p["ep_min"]
                     or ratio[a:b + 1].max() > p["ep_fuerte"] * th_rat]

    if manual is not None:
        frame_guess, origen = int(manual), "manual"
    elif episodios:
        fuerza = [float(np.sum(ratio[a:b + 1] - th_rat)) for a, b in episodios]
        mejor = int(np.argmax(fuerza))
        frame_guess = int(fr_idx[episodios[mejor][0]])
        origen = "inicio del episodio mas fuerte"
    else:
        if verbose:
            print(f"\n{'frame':>7} {'activ':>8} {'ratio':>8}  motivo")
            for i in np.argsort(ratio)[::-1][:15]:
                mv = ",".join(([] if ratio[i] > th_rat else ["no localizado"]) +
                              ([] if act[i] > th_act else ["intensidad baja"]))
                print(f"{fr_idx[i]:>7} {act[i]:>8.2f} {ratio[i]:>8.2f}  "
                      f"{mv or '(pasa ambos gates - revisa ep_min)'}")
        raise RuntimeError(
            f"Sin episodios localizados (ratio>{th_rat:.2f} y act>{th_act:.2f}). "
            f"Revisa la tabla de arriba o pasa manual=<frame>.")

    lim = max(p["win"], frame_guess - p["margen"] - 1)
    k = min(p["win"], lim)
    peor = np.array([act[i:i + k].max() for i in range(max(1, lim - k + 1))])
    tol = p["tol"] * base
    ok = np.where(peor <= tol)[0]
    if len(ok):
        iq = int(ok[-1])
        crit = f"mas cercana al onset con peor actividad <= {tol:.2f}"
    else:
        iq = int(np.argmin(peor))
        crit = "ninguna bajo tolerancia -> la mas quieta"
    piso_ini, piso_fin = int(fr_idx[iq]), int(fr_idx[iq] + k)

    res = dict(frame_guess=frame_guess, origen_onset=origen,
               piso_ini=piso_ini, piso_fin=piso_fin, criterio_piso=crit,
               th_ratio=th_rat, th_actividad=th_act, ratio_basal=rbase,
               actividad_basal=base, peor_actividad=float(peor[iq]),
               bajo_tolerancia=bool(len(ok)),
               episodios=[[int(fr_idx[a]), int(fr_idx[b]),
                           float(ratio[a:b + 1].max()),
                           float(np.sum(ratio[a:b + 1] - th_rat))]
                          for a, b in episodios])

    if verbose:
        print(f"\n[escaneo] {len(act)} frames | grilla {p['grid'][0]}x{p['grid'][1]}")
        print(f"[escaneo] ratio basal {rbase:.2f} -> umbral localizacion "
              f"{th_rat:.2f} | {int(loc.sum())} frames localizados")
        print(f"[escaneo] actividad basal (fondo NO localizado) {base:.2f} "
              f"-> umbral {th_act:.2f}")
        print(f"[escaneo] {len(episodios)} episodio(s):")
        for a, b in episodios:
            f_ = float(np.sum(ratio[a:b + 1] - th_rat))
            mk = "  <- ELEGIDO" if fr_idx[a] == frame_guess else ""
            ex = " (corto pero intenso)" if (b - a + 1) < p["ep_min"] else ""
            print(f"            [f{fr_idx[a]:>4}, f{fr_idx[b]:>4}] {b-a+1:>3} "
                  f"frames | ratio max {ratio[a:b+1].max():>5.2f} | "
                  f"fuerza {f_:>7.1f}{ex}{mk}")
        print(f"[escaneo] FRAME_GUESS = {frame_guess} ({origen})")
        print(f"[escaneo] ventana del piso: [f{piso_ini}, f{piso_fin}) "
              f"| peor act = {peor[iq]:.2f} | {crit}")
        sep = frame_guess - piso_fin
        print(f"[escaneo] separacion piso->onset: {sep} frames "
              f"({sep / max(fps, 1e-6):.1f} s)")
        if not len(ok):
            print("   AVISO: ninguna ventana bajo tolerancia. El fondo limpio "
                  "saldra de un tramo con actividad; vigila el chequeo de "
                  "registro de la Fase 1.")
        if sep > 5 * fps:
            print("   AVISO: mas de 5 s entre el fondo y la detonacion. El "
                  "registro ECC tendra que absorber mas deriva del dron.")
    return res


def grafico_actividad(sc: dict, res: dict, ruta: str | None = None):
    """Las dos curvas del escaneo con episodios, onset y ventana tranquila."""
    import matplotlib.pyplot as plt
    act, bg = np.asarray(sc["act"]), np.asarray(sc["bg"])
    fr_idx = np.arange(1, len(act) + 1)
    ratio = act / np.maximum(bg, 1e-3)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(fr_idx, act, lw=0.9, color="crimson", label="max por bloque")
    ax[0].plot(fr_idx, bg, lw=0.8, color="steelblue", alpha=0.7,
               label="mediana bloques")
    ax[0].axhline(res["th_actividad"], color="k", ls="--", lw=0.8,
                  label=f"umbral {res['th_actividad']:.1f}")
    ax[0].set_ylabel("actividad")
    ax[1].plot(fr_idx, ratio, lw=0.9, color="darkgreen")
    ax[1].axhline(res["th_ratio"], color="k", ls="--", lw=0.8,
                  label=f"umbral {res['th_ratio']:.1f}")
    ax[1].set_ylabel("localizacion")
    ax[1].set_xlabel("Frame")
    for a_ in ax:
        for e in res["episodios"]:
            a_.axvspan(e[0], e[1], color="red", alpha=0.18)
        a_.axvspan(res["piso_ini"], res["piso_fin"], color="green", alpha=0.3,
                   label="fondo limpio")
        a_.axvline(res["frame_guess"], color="b", ls=":", lw=1.2,
                   label=f"onset f{res['frame_guess']}")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=7)
    plt.suptitle("Episodios localizados (rojo) y ventanas de calibracion")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=120)
    plt.show()


def guardar_geometria(geo: dict, ruta: str) -> str:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(geo, fh, indent=2, ensure_ascii=False)
    print(f"[geo] -> {ruta_corta(ruta)}")
    return ruta


def cargar_geometria(ruta: str) -> dict:
    with open(ruta, "r", encoding="utf-8") as fh:
        return json.load(fh)


PAR_AIRBLAST = dict(
    temp_aire_c=15.0,
    pre_roll=30,
    ventana=120,
    cutoff_hz=5.0,
    prom_factor=2.5,
    piso_cap_pctl=50,
    blur_pre=1.5,
    th_mask_ini=2.0,
    niveles_th=0.25,
    n_chunks_x=8,
    gain_fit=4.5,
    n_estela=6,
    k_vecinos=9,
    min_vecinos=12,
    cronico_pct=0.35,
    rb_nbins=400,
    k_arco=4.0,
    centro=None,
)


def _velocidad_sonido(temp_c: float) -> float:
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def precomputo_airblast(cfg, onset: dict, par=None, verbose=True) -> dict:
    """Mapa de prominencia + filtros espaciales. Automatico y caro."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as e:
        raise ImportError(
            "La calibracion del airblast necesita scipy (scipy.signal.butter). "
            "Instalalo con: pip install scipy") from e

    p = dict(PAR_AIRBLAST)
    p.update(par or {})
    vid = propiedades_video(cfg.video)
    fps, n_frames = vid["fps"], vid["n_frames"]
    w_nat, h_nat = vid["ancho"], vid["alto"]

    guess = int(onset["frame_guess"])
    piso_ini, piso_fin = int(onset["piso_ini"]), int(onset["piso_fin"])
    c_sonido = _velocidad_sonido(p["temp_aire_c"])
    if verbose:
        print(f"[acustica] c = {c_sonido:.1f} m/s @ {p['temp_aire_c']:.1f} C")

    f_ini = max(0, guess - p["pre_roll"])
    t_ab = min(p["ventana"], n_frames - f_ini)
    if t_ab < p["pre_roll"] + 20:
        raise RuntimeError(f"Ventana de solo {t_ab} frames; revisa el onset "
                           f"(f{guess}) o baja pre_roll.")
    idx_real = list(range(f_ini, f_ini + t_ab))
    t_piso = piso_fin - piso_ini
    if piso_fin > f_ini and verbose:
        print(f"   AVISO: la ventana del piso [f{piso_ini}, f{piso_fin}) se "
              f"solapa con la de anotacion (inicia en f{f_ini}). Revisa la "
              f"Fase 0a.")
    if verbose:
        print(f"[ventana] anotacion {t_ab} frames [f{f_ini}, {f_ini+t_ab}) @ "
              f"{w_nat}x{h_nat}")
        print(f"[ventana] piso de ruido {t_piso} frames [f{piso_ini}, {piso_fin})")

    def _gris(fr, xa, xb):
        """Gris del recorte de columnas. El blur va ANTES del recorte para no introducir un
        borde artificial en la frontera del chunk.
        """
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if p["blur_pre"] and p["blur_pre"] > 0:
            g = cv2.GaussianBlur(g, (0, 0), p["blur_pre"])
        return g[:, xa:xb]

    ruta_prom = os.path.join(
        cfg.dir_cache,
        f"prom_nat_{os.path.splitext(os.path.basename(cfg.video))[0]}.dat")
    os.makedirs(cfg.dir_cache, exist_ok=True)
    if verbose:
        print(f"[prominencia] {p['n_chunks_x']} chunks, {p['cutoff_hz']} Hz, "
              f"factor {p['prom_factor']}, tope p{p['piso_cap_pctl']}, "
              f"blur s={p['blur_pre']} -> memmap "
              f"{t_ab * h_nat * w_nat / 1e6:.0f} MB")
    b_hp, a_hp = butter(2, p["cutoff_hz"] / (0.5 * fps), btype="high")
    prom = np.memmap(ruta_prom, dtype=np.uint8, mode="w+",
                     shape=(t_ab, h_nat, w_nat))
    k3 = np.ones((3, 3), np.uint8)
    bordes = np.linspace(0, w_nat, p["n_chunks_x"] + 1, dtype=int)
    pisos = []

    for ci in range(p["n_chunks_x"]):
        xa, xb = bordes[ci], bordes[ci + 1]

        capq = cv2.VideoCapture(cfg.video)
        capq.set(cv2.CAP_PROP_POS_FRAMES, piso_ini)
        sq = []
        for _ in range(t_piso):
            ok, fr = capq.read()
            if not ok:
                break
            sq.append(_gris(fr, xa, xb))
        capq.release()
        if len(sq) < 5:
            raise RuntimeError(f"Solo {len(sq)} frames leidos de la ventana "
                               f"tranquila [f{piso_ini}, f{piso_fin}).")
        Vq = np.asarray(sq, np.float32)
        del sq
        piso = np.std(Vq, axis=0).astype(np.float32) * p["prom_factor"]
        piso = np.minimum(piso, np.percentile(piso, p["piso_cap_pctl"])
                          ).astype(np.float32)
        pisos.append(float(piso.max()))
        del Vq
        gc.collect()

        cap = cv2.VideoCapture(cfg.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_ini)
        sl = []
        for _ in range(t_ab):
            ok, fr = cap.read()
            if not ok:
                break
            sl.append(_gris(fr, xa, xb))
        cap.release()
        V = np.asarray(sl, dtype=np.float32)
        del sl

        filt = np.abs(filtfilt(b_hp, a_hp, V.astype(np.float64),
                               axis=0)).astype(np.float32)
        del V
        gc.collect()

        core = filt[2:t_ab - 2]
        es_max = ((core > filt[1:t_ab - 3]) & (core > filt[3:t_ab - 1]) &
                  (core > filt[0:t_ab - 4]) & (core > filt[4:t_ab]))
        valido = es_max & (core > piso)
        pc = np.zeros_like(core)
        pc[valido] = core[valido]
        del filt, es_max, valido, piso, core
        gc.collect()

        for t in range(pc.shape[0]):
            q = np.clip(pc[t], 0, 255).astype(np.uint8)
            prom[t + 2, :, xa:xb] = cv2.dilate(q, k3, iterations=1)
        del pc
        gc.collect()
        if verbose:
            print(f"   chunk {ci+1}/{p['n_chunks_x']} [x {xa}:{xb}] ok")
    prom.flush()
    if verbose:
        print(f"   piso acotado: max por chunk {min(pisos):.1f}-{max(pisos):.1f}")

    ocup_pre = float(np.mean([(prom[t] > p["th_mask_ini"]).mean()
                              for t in range(2, p["pre_roll"])]))
    ocup_post = float(np.mean([(prom[t] > p["th_mask_ini"]).mean()
                               for t in range(p["pre_roll"],
                                              min(p["pre_roll"] + 20, t_ab - 2))]))
    if verbose:
        print(f"\n[diagnostico] ocupacion pre-detonacion  : {ocup_pre*100:.2f}%")
        print(f"[diagnostico] ocupacion post-detonacion : {ocup_post*100:.2f}%  "
              f"(contraste {ocup_post/max(ocup_pre,1e-9):.1f}x)")
        if ocup_pre > 0.05:
            print("   AVISO: ocupacion PRE alta. Es esperable con prom_factor "
                  "bajo; para eso estan los filtros 1-4.")
        if ocup_post / max(ocup_pre, 1e-9) < 3:
            print("   AVISO: contraste bajo, la detonacion apenas destaca del "
                  "fondo. Revisa el onset antes de anotar.")

    if p["centro"] is None:
        capc = cv2.VideoCapture(cfg.video)
        capc.set(cv2.CAP_PROP_POS_FRAMES, max(0, guess - 1))
        g1 = cv2.cvtColor(capc.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(capc.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        capc.release()
        dd = cv2.GaussianBlur(np.abs(g2 - g1), (0, 0), 9)
        ys, xs = np.where(dd >= np.percentile(dd, 99.9))
        cx_ab, cy_ab = float(xs.mean()), float(ys.mean())
        del g1, g2, dd
        gc.collect()
        if verbose:
            print(f"[arco] centro automatico: ({cx_ab:.0f}, {cy_ab:.0f})")
    else:
        cx_ab, cy_ab = map(float, p["centro"])
        if verbose:
            print(f"[arco] centro manual: ({cx_ab:.0f}, {cy_ab:.0f})")

    yy, xx = np.mgrid[0:h_nat, 0:w_nat]
    rmax = float(np.hypot(max(cx_ab, w_nat - cx_ab), max(cy_ab, h_nat - cy_ab)))
    rbin = np.clip((np.hypot(xx - cx_ab, yy - cy_ab) / rmax * p["rb_nbins"])
                   .astype(np.int32), 0, p["rb_nbins"] - 1)
    del yy, xx
    gc.collect()
    pob_rbin = np.maximum(np.bincount(rbin.ravel(), minlength=p["rb_nbins"]), 1)
    if verbose:
        print(f"[arco] {p['rb_nbins']} anillos de {rmax/p['rb_nbins']:.1f} px")

    acum = np.zeros((h_nat, w_nat), np.uint16)
    for t in range(2, p["pre_roll"]):
        acum += (prom[t] > p["th_mask_ini"]).astype(np.uint16)
    cronicos = acum > (p["cronico_pct"] * max(1, p["pre_roll"] - 2))
    del acum
    gc.collect()
    if verbose:
        print(f"[filtro] pixeles cronicos: {cronicos.mean()*100:.2f}% del cuadro")

    return dict(par=p, prom=prom, ruta_prom=ruta_prom, idx_real=idx_real,
                t_ab=t_ab, w_nat=w_nat, h_nat=h_nat, fps=fps,
                f_ini=f_ini, frame_guess=guess,
                piso_ini=piso_ini, piso_fin=piso_fin,
                cronicos=cronicos, rbin=rbin, pob_rbin=pob_rbin,
                centro_arco=(cx_ab, cy_ab), c_sonido=c_sonido,
                ocup_pre=ocup_pre, ocup_post=ocup_post, video=cfg.video)


def anotar_airblast(ctx: dict) -> dict:
    """GUI de anotacion del frente de choque. Es la parte MANUAL."""
    p = ctx["par"]
    prom, idx_real, t_ab = ctx["prom"], ctx["idx_real"], ctx["t_ab"]
    w_nat, h_nat = ctx["w_nat"], ctx["h_nat"]
    cronicos, rbin, pob_rbin = ctx["cronicos"], ctx["rbin"], ctx["pob_rbin"]
    cx_ab, cy_ab = ctx["centro_arco"]
    nbins = p["rb_nbins"]

    fit = min(1280 / w_nat, 720 / h_nat)
    w_f, h_f = int(w_nat * fit), int(h_nat * fit)
    st = {"clicks": {}, "i": min(p["pre_roll"], t_ab - 3), "vista": 0,
          "estela": True, "th": float(p["th_mask_ini"]), "fit": True,
          "cx": w_nat // 2, "cy": h_nat // 2, "dirty": True,
          "f_vec": True, "f_cron": True, "f_pers": False, "f_arco": True,
          "minv": p["min_vecinos"], "karc": p["k_arco"], "rapido": False}
    MODOS = ["MASK", "OVERLAY", "RGB"]

    kv = np.ones((p["k_vecinos"], p["k_vecinos"]), np.float32)
    cron_f = cv2.resize(cronicos.astype(np.uint8), (w_f, h_f),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    rbin_f = cv2.resize(rbin, (w_f, h_f), interpolation=cv2.INTER_NEAREST)
    pob_f = np.maximum(np.bincount(rbin_f.ravel(), minlength=nbins), 1)
    kf = max(3, int(round(p["k_vecinos"] * fit)) | 1)
    kvf = np.ones((kf, kf), np.float32)
    esc_v = (kf / p["k_vecinos"]) ** 2

    cache_capa, cache_rgb = {}, {}

    def _vp():
        x0 = int(np.clip(st["cx"] - w_f // 2, 0, max(0, w_nat - w_f)))
        y0 = int(np.clip(st["cy"] - h_f // 2, 0, max(0, h_nat - h_f)))
        return x0, y0

    def _clave(fi):
        return (fi, round(st["th"], 3), st["f_vec"], st["f_cron"], st["f_pers"],
                st["f_arco"], st["minv"], round(st["karc"], 2), st["fit"],
                st["cx"] if not st["fit"] else 0,
                st["cy"] if not st["fit"] else 0)

    def _capa(fi, th):
        k = _clave(fi)
        c = cache_capa.get(k)
        if c is not None:
            return c
        if st["fit"]:
            m = cv2.resize((prom[fi] > th).astype(np.uint8), (w_f, h_f),
                           interpolation=cv2.INTER_AREA) > 0
            if st["f_pers"] and fi >= 3:
                m = m | (cv2.resize((prom[fi - 1] > th).astype(np.uint8),
                                    (w_f, h_f),
                                    interpolation=cv2.INTER_AREA) > 0)
            cron, rb, pob = cron_f, rbin_f, pob_f
            kk, minv = kvf, max(1, int(st["minv"] * esc_v))
        else:
            x0, y0 = _vp()
            m = prom[fi][y0:y0 + h_f, x0:x0 + w_f] > th
            if st["f_pers"] and fi >= 3:
                m = m | (prom[fi - 1][y0:y0 + h_f, x0:x0 + w_f] > th)
            cron = cronicos[y0:y0 + h_f, x0:x0 + w_f]
            rb, pob = rbin[y0:y0 + h_f, x0:x0 + w_f], pob_rbin
            kk, minv = kv, st["minv"]
        if st["f_cron"]:
            m = m & (~cron)
        if st["f_vec"]:
            dens = cv2.filter2D(m.astype(np.float32), -1, kk,
                                borderType=cv2.BORDER_CONSTANT)
            m = m & (dens >= minv)
        if st["f_arco"] and m.any():
            cnt = np.bincount(rb[m], minlength=nbins)
            ok = (cnt / pob) >= (st["karc"] * max(float(m.mean()), 1e-6))
            m = m & ok[rb]
        out = m.astype(np.uint8) * 255
        if st["fit"]:
            out = np.clip(out.astype(np.float32) * (p["gain_fit"] * 0.6),
                          0, 255).astype(np.uint8)
        if len(cache_capa) > 260:
            cache_capa.clear()
        cache_capa[k] = out
        return out

    def _cb(ev, x, y, flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            fr = idx_real[st["i"]]
            nx, ny = (x / fit, y / fit) if st["fit"] else \
                     (_vp()[0] + x, _vp()[1] + y)
            st["clicks"].setdefault(fr, []).append((nx, ny))
            st["dirty"] = True
            print(f"  f{fr}: x={nx:.1f} y={ny:.1f}")
        elif ev == cv2.EVENT_RBUTTONDOWN:
            st["cx"], st["cy"] = (x / fit, y / fit) if st["fit"] else \
                                 (_vp()[0] + x, _vp()[1] + y)
            st["fit"] = False
            st["dirty"] = True
            cache_capa.clear()

    def _rgb_nativo(fi):
        capr = cv2.VideoCapture(ctx["video"])
        capr.set(cv2.CAP_PROP_POS_FRAMES, idx_real[fi])
        ok, fr = capr.read()
        capr.release()
        return fr if ok else np.zeros((h_nat, w_nat, 3), np.uint8)

    WIN = "Calibrador cinematico de onda de choque"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, w_f, h_f)
    cv2.setMouseCallback(WIN, _cb)
    print("[teclas] a/d +-1  w/s +-10  f=1:1  m=vista  t=estela  +/-=umbral")
    print("[teclas] 1=vecinos 2=cronicos 3=persistencia 4=arco  [ ]=minv  ; '=Karco")
    print("[teclas] r=modo rapido  z=deshacer  Enter o q = terminar")

    while True:
        if st["dirty"]:
            t_ini = time.time()
            i, th = st["i"], st["th"]
            fr = idx_real[i]
            cap_m = _capa(i, th)
            con_estela = st["estela"] and not st["rapido"]

            if st["vista"] in (1, 2):
                if i not in cache_rgb:
                    if len(cache_rgb) > 24:
                        cache_rgb.clear()
                    cache_rgb[i] = _rgb_nativo(i)
                base = cache_rgb[i]
                if st["fit"]:
                    base = cv2.resize(base, (w_f, h_f),
                                      interpolation=cv2.INTER_AREA)
                else:
                    x0, y0 = _vp()
                    base = base[y0:y0 + h_f, x0:x0 + w_f]
                disp = base.copy() if st["vista"] == 2 \
                    else (base * 0.45).astype(np.uint8)
                if st["vista"] == 1:
                    if con_estela:
                        for j in range(p["n_estela"], 0, -1):
                            if i - j >= 2:
                                disp[_capa(i - j, th) > 0] = (
                                    0, max(0, 90 - 12 * j), max(0, 160 - 20 * j))
                    disp[cap_m > 0] = (0, 255, 255)
            else:
                disp = np.zeros((h_f, w_f, 3), np.uint8)
                if con_estela:
                    for j in range(p["n_estela"], 0, -1):
                        if i - j >= 2:
                            v = int(max(30, 210 - 28 * j))
                            disp[_capa(i - j, th) > 0] = (v, v, v)
                for c in range(3):
                    disp[..., c] = np.maximum(disp[..., c], cap_m)

            t_ref = min(st["clicks"]) if st["clicks"] else None
            modo = "CENTRO" if (t_ref is None or fr <= t_ref) else "FRENTE"
            flt = ("V" if st["f_vec"] else "-") + ("C" if st["f_cron"] else "-") \
                + ("P" if st["f_pers"] else "-") + ("A" if st["f_arco"] else "-")
            oc = float((cap_m > 0).mean()) * 100
            ms = (time.time() - t_ini) * 1000
            cv2.putText(disp, f"f{fr} | {modo} | {MODOS[st['vista']]} | "
                              f"{'AJUST' if st['fit'] else '1:1'} | th={th:.2f} | "
                              f"[{flt}] minv={st['minv']} Ka={st['karc']:.1f} | "
                              f"ocup {oc:.2f}% | {ms:.0f} ms"
                              f"{' | RAPIDO' if st['rapido'] else ''} | "
                              f"{sum(len(v) for v in st['clicks'].values())} pts",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2)
            if st["fit"]:
                cv2.drawMarker(disp, (int(cx_ab * fit), int(cy_ab * fit)),
                               (255, 200, 0), cv2.MARKER_TILTED_CROSS, 18, 1)
            for f_, pts in st["clicks"].items():
                if f_ != fr:
                    continue
                col = (255, 128, 0) if f_ == t_ref else (0, 0, 255)
                for px, py in pts:
                    if st["fit"]:
                        vx, vy = px * fit, py * fit
                    else:
                        x0, y0 = _vp()
                        vx, vy = px - x0, py - y0
                    cv2.drawMarker(disp, (int(vx), int(vy)), col,
                                   cv2.MARKER_CROSS, 16, 2)
            cv2.imshow(WIN, disp)
            st["dirty"] = False

        k = cv2.waitKey(15) & 0xFF
        if k == 255:
            continue
        st["dirty"] = True
        reset = False
        if k in (13, ord("q"), ord("Q")):
            break
        elif k in (ord("d"), ord("D")):
            st["i"] = min(t_ab - 3, st["i"] + 1)
        elif k in (ord("a"), ord("A")):
            st["i"] = max(2, st["i"] - 1)
        elif k in (ord("s"), ord("S")):
            st["i"] = min(t_ab - 3, st["i"] + 10)
        elif k in (ord("w"), ord("W")):
            st["i"] = max(2, st["i"] - 10)
        elif k in (ord("f"), ord("F")):
            st["fit"] = not st["fit"]; reset = True
        elif k in (ord("m"), ord("M")):
            st["vista"] = (st["vista"] + 1) % 3
        elif k in (ord("t"), ord("T")):
            st["estela"] = not st["estela"]
        elif k in (ord("r"), ord("R")):
            st["rapido"] = not st["rapido"]
        elif k == ord("1"):
            st["f_vec"] = not st["f_vec"]; reset = True
        elif k == ord("2"):
            st["f_cron"] = not st["f_cron"]; reset = True
        elif k == ord("3"):
            st["f_pers"] = not st["f_pers"]; reset = True
        elif k == ord("4"):
            st["f_arco"] = not st["f_arco"]; reset = True
        elif k == ord("]"):
            st["minv"] = min(p["k_vecinos"] ** 2, st["minv"] + 2); reset = True
        elif k == ord("["):
            st["minv"] = max(1, st["minv"] - 2); reset = True
        elif k == ord("'"):
            st["karc"] = min(50.0, st["karc"] + 0.5); reset = True
        elif k == ord(";"):
            st["karc"] = max(1.0, st["karc"] - 0.5); reset = True
        elif k in (ord("+"), ord("=")):
            st["th"] = min(254.0, st["th"] + p["niveles_th"]); reset = True
        elif k in (ord("-"), ord("_")):
            st["th"] = max(0.25, st["th"] - p["niveles_th"]); reset = True
        elif k in (ord("z"), ord("Z")):
            fr = idx_real[st["i"]]
            if fr in st["clicks"] and st["clicks"][fr]:
                st["clicks"][fr].pop()
                if not st["clicks"][fr]:
                    del st["clicks"][fr]
        if reset:
            cache_capa.clear()
    cv2.destroyAllWindows()
    cache_rgb.clear()
    cache_capa.clear()
    return st


def ajustar_airblast(st: dict, ctx: dict, verbose=True) -> dict:
    """Regresion radio-vs-tiempo de intercepto libre -> escala y t0 subframe."""
    clicks = st["clicks"]
    if len(clicks) < 3:
        raise RuntimeError(f"Se necesitan clics en >=3 frames distintos; "
                           f"hay {len(clicks)}.")
    p, fps = ctx["par"], ctx["fps"]
    t_ref = min(clicks)
    x0c, y0c = clicks[t_ref][0]
    dt_l, r_l = [], []
    for f_, pts in clicks.items():
        if f_ == t_ref:
            continue
        for x, y in pts:
            dt_l.append(f_ - t_ref)
            r_l.append(np.hypot(x - x0c, y - y0c))
    dt_arr, r_arr = np.array(dt_l, float), np.array(r_l, float)
    if len(dt_arr) < 3:
        raise RuntimeError("Menos de 3 puntos de frente fuera del frame de "
                           "referencia: no se puede ajustar.")

    slope_px_f, intercept = np.polyfit(dt_arr, r_arr, 1)
    if slope_px_f <= 0:
        raise RuntimeError(f"Pendiente no positiva ({slope_px_f:.2f} px/frame): "
                           "el radio no crece con el tiempo. Revisa que el "
                           "primer frame anotado sea el CENTRO.")
    r2 = 1 - np.sum((r_arr - (slope_px_f * dt_arr + intercept)) ** 2) \
        / np.sum((r_arr - r_arr.mean()) ** 2)
    dt_corr = -intercept / slope_px_f
    t0 = float(t_ref + dt_corr)
    ppm_nat = (slope_px_f * fps) / ctx["c_sonido"]

    cal = dict(T0_SUBFRAME=t0, PIVOT_X_NAT=float(x0c), PIVOT_Y_NAT=float(y0c),
               PPM_NAT=float(ppm_nat), SLOPE_PX_FRAME=float(slope_px_f),
               R2=float(r2), T_REF=int(t_ref), DT_CORR=float(dt_corr),
               C_SONIDO=float(ctx["c_sonido"]), TEMP_C=float(p["temp_aire_c"]),
               N_PUNTOS=int(len(dt_arr)), TH_USADO=float(st["th"]),
               OCUP_PRE=float(ctx["ocup_pre"]), OCUP_POST=float(ctx["ocup_post"]),
               PISO_INI=int(ctx["piso_ini"]), PISO_FIN=int(ctx["piso_fin"]),
               PISO_CAP_PCTL=int(p["piso_cap_pctl"]),
               PROM_FACTOR=float(p["prom_factor"]), BLUR_PRE=float(p["blur_pre"]),
               FILTRO_VEC=bool(st["f_vec"]), FILTRO_CRON=bool(st["f_cron"]),
               FILTRO_PERS=bool(st["f_pers"]), FILTRO_ARCO=bool(st["f_arco"]),
               MIN_VECINOS=int(st["minv"]), K_ARCO=float(st["karc"]),
               K_VECINOS=int(p["k_vecinos"]), CRONICO_PCT=float(p["cronico_pct"]),
               CENTRO_ARCO=tuple(float(v) for v in ctx["centro_arco"]),
               _ajuste=dict(dt=dt_arr.tolist(), r=r_arr.tolist(),
                            intercept=float(intercept)))
    if verbose:
        print(f"\n{'='*62}")
        print(f"  t0 sub-frame : {t0:.2f}  (pivote f{int(round(t0))}, "
              f"ref f{t_ref}, corr {dt_corr:+.2f})")
        print(f"  Ajuste       : R2={r2:.4f} con {len(dt_arr)} puntos "
              f"| c={ctx['c_sonido']:.1f} m/s")
        print(f"  Escala       : {ppm_nat:.2f} px/m nativo")
        print(f"  GSD nativo   : {1.0/ppm_nat:.5f} m/px")
        print(f"  Pivote nat   : ({x0c:.0f}, {y0c:.0f})")
        print(f"{'='*62}")
        if r2 < 0.90:
            print("   AVISO: R2 bajo. A 30 fps una onda que cruza el cuadro en "
                  "2 frames deja 1-2 muestras utiles y la regresion se vuelve "
                  "erratica. Anota mas frames o desconfia de esta escala.")
    return cal


def grafico_airblast(cal: dict, ruta: str | None = None):
    """La nube de puntos radio-vs-tiempo y la recta ajustada."""
    import matplotlib.pyplot as plt
    aj = cal.get("_ajuste")
    if not aj:
        print("La calibracion no trae los puntos del ajuste (es de V7.2).")
        return
    dt, r = np.asarray(aj["dt"]), np.asarray(aj["r"])
    m, b = cal["SLOPE_PX_FRAME"], aj["intercept"]
    plt.figure(figsize=(8, 4))
    plt.scatter(dt, r, s=18, label="observaciones")
    tp = np.linspace(cal["DT_CORR"], dt.max(), 100)
    plt.plot(tp, m * tp + b, "r--", label=f"ajuste (R2={cal['R2']:.3f})")
    plt.axhline(0, lw=0.5, color="k")
    plt.axvline(cal["DT_CORR"], ls=":", color="orange",
                label=f"t0 corr {cal['DT_CORR']:+.2f}")
    plt.xlabel(f"delta frames desde f{cal['T_REF']}")
    plt.ylabel("Radio del frente (px nativos)")
    plt.title("Expansion acustica - intercepto libre")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=120)
    plt.show()


def guardar_calibracion(cal: dict, ruta: str, video_nombre: str | None = None) -> str:
    """Guarda el pkl y le sella a que video pertenece."""
    cal = dict(cal)
    if video_nombre:
        cal["VIDEO"] = os.path.basename(video_nombre)
    if video_nombre:
        esperado = nombre_calibracion(video_nombre)
        if os.path.basename(ruta) != esperado:
            print(f"[calibracion] [aviso] se esta guardando como "
                  f"{os.path.basename(ruta)}, no como {esperado}. El nombre "
                  f"canonico es el que permite tener varios videos en datos/ "
                  f"sin ambiguedad.")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as fh:
        pickle.dump(cal, fh)
    print(f"[calibracion] -> {ruta_corta(ruta)}"
          + (f"   (sellada para {cal['VIDEO']})" if video_nombre else ""))
    return ruta


def calibrar_airblast(cfg, onset: dict, par=None, forzar=False,
                      verbose=True) -> dict:
    """Calibracion completa: precomputo -> anotacion manual -> ajuste -> .pkl."""
    canonico = nombre_calibracion(cfg.video_nombre)

    migrar_nombre_calibracion(cfg, verbose=verbose)

    reusar = None
    if cfg.calibracion_nombre and cfg.calibracion_nombre != "auto" \
            and os.path.exists(cfg.calibracion):
        calza = calibracion_calza(cfg.calibracion, cfg.video_nombre)
        if calza is False:
            print(f"[calibracion] {os.path.basename(cfg.calibracion)} es de OTRO "
                  f"video: NO se usa para {os.path.basename(cfg.video_nombre)}.")
            cfg.calibracion_nombre = "auto"
        else:
            reusar = cfg.calibracion
            if calza is None and verbose:
                print(f"[calibracion] {os.path.basename(cfg.calibracion)} no "
                      f"declara video por dentro; se usa porque esta fijada a "
                      f"mano en la config.")
    if reusar is None and os.path.exists(os.path.join(cfg.dir_fase(0), canonico)):
        reusar = os.path.join(cfg.dir_fase(0), canonico)

    if reusar and not forzar:
        cfg.calibracion_nombre = os.path.basename(reusar)
        cal = cargar_calibracion(reusar)
        if verbose:
            print(f"[calibracion] {os.path.basename(reusar)} ya existe: "
                  f"no se vuelve a anotar (forzar=True para rehacerla).")
        return cal

    if verbose:
        print(f"[calibracion] se guardara como {canonico}")
    ctx = precomputo_airblast(cfg, onset, par=par, verbose=verbose)
    try:
        st = anotar_airblast(ctx)
        cal = ajustar_airblast(st, ctx, verbose=verbose)
    finally:
        prom = ctx.pop("prom", None)
        del prom
        gc.collect()
    cfg.calibracion_nombre = canonico
    guardar_calibracion(cal, cfg.calibracion, video_nombre=cfg.video_nombre)
    return cal


PREPROC_DEFECTO = dict(clahe_clip=2.0, clahe_tile=8, gamma=1.2,
                       saturation=1.3, contrast=1.0, brightness=0,
                       unsharp_amount=0.3)

BG_WINDOW_DEFECTO = 40


def preprocesar_frame(frame, clahe_clip=0.0, clahe_tile=8, gamma=1.0,
                      saturation=1.0, contrast=1.0, brightness=0,
                      unsharp_amount=0.0):
    """CLAHE -> gamma -> saturacion -> contraste/brillo -> unsharp."""
    out = frame.copy()
    if clahe_clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_tile, clahe_tile))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if gamma != 1.0:
        lut = np.array([min(255, int(255 * (i / 255.0) ** (1.0 / gamma)))
                        for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)
    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if contrast != 1.0 or brightness != 0:
        out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)
    if unsharp_amount > 0:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
        out = cv2.addWeighted(out, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
    return out


def preprocesador(cfg: dict | None = None):
    """Devuelve un callable frame -> frame con la config congelada."""
    cfg = PREPROC_DEFECTO if cfg is None else cfg
    return lambda f: preprocesar_frame(f, **cfg)


def magnitud_gradiente(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(np.sqrt(gx ** 2 + gy ** 2))


def modelo_fondo(ruta_video, pivot_frame, escala=0.5, preproc=None,
                 dir_salida=None, noise_floor=1.0, bg_window=BG_WINDOW_DEFECTO,
                 step=1, blur_ksize=(5, 5)):
    """Mediana temporal pre-tronadura + mapa de ruido robusto (MAD)."""
    if not os.path.exists(ruta_video):
        raise FileNotFoundError(f"No se encontro: {ruta_video}")
    cap = cv2.VideoCapture(ruta_video)
    frames_gray, frames_color = [], []
    inicio = 0 if bg_window is None else max(0, pivot_frame - bg_window)
    n_leer = pivot_frame - inicio
    if inicio > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    print(f"Fondo con frames [{inicio}, {pivot_frame}) paso {step} "
          f"(escala {escala}, blur={blur_ksize})...")
    for k in range(n_leer):
        ok, frame = cap.read()
        if not ok:
            break
        if step > 1 and (k % step) != 0:
            continue
        small = cv2.resize(frame, None, fx=escala, fy=escala)
        if preproc is not None:
            small = preproc(small)
        if blur_ksize is not None:
            small = cv2.GaussianBlur(small, blur_ksize, 0)
        frames_color.append(small)
        frames_gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames_gray:
        raise RuntimeError("No se pudieron leer frames del video.")

    pila = np.stack(frames_gray, axis=0).astype(np.float32)
    bg_gray = np.median(pila, axis=0)
    bg_color = np.median(frames_color, axis=0).astype(np.uint8)
    mad = np.median(np.abs(pila - bg_gray), axis=0)
    noise = np.maximum(1.4826 * mad, noise_floor).astype(np.float32)
    bg_gray = bg_gray.astype(np.uint8)

    if dir_salida is not None:
        os.makedirs(dir_salida, exist_ok=True)
        cv2.imwrite(os.path.join(dir_salida, "bg_gray.png"), bg_gray)
        cv2.imwrite(os.path.join(dir_salida, "bg_color.png"), bg_color)
    print(f"Ruido robusto (MAD) promedio: {np.mean(noise):.2f} niveles")
    return bg_gray, bg_color, noise


def guardar_fondo(ruta, bg_gray, bg_color, noise, meta=None):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.savez_compressed(ruta, bg_gray=bg_gray, bg_color=bg_color, noise=noise,
                        meta=np.array([str(meta or {})], dtype=object))
    print(f"[fase0] fondo -> {ruta_corta(ruta)}")
    return ruta


def cargar_fondo(ruta, forma=None):
    """Devuelve (bg_gray, bg_color, noise, grad_bg) redimensionados a `forma`."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta el modelo de fondo: {ruta}. "
                                "Corre el notebook de la Fase 0.")
    z = np.load(ruta, allow_pickle=True)
    bg_gray, bg_color, noise = z["bg_gray"], z["bg_color"], z["noise"]
    if forma is not None and bg_gray.shape != forma:
        print(f"  aviso: fondo {bg_gray.shape} -> resize a {forma}")
        bg_gray = cv2.resize(bg_gray, (forma[1], forma[0]))
        noise = cv2.resize(noise, (forma[1], forma[0]))
        bg_color = cv2.resize(bg_color, (forma[1], forma[0]))
    return bg_gray, bg_color, noise, magnitud_gradiente(bg_gray)


def extraer_frames_jpeg(ruta_video, frames, carpeta, escala=0.5):
    """Extrae `frames` a JPEGs 0000.jpg... Devuelve {indice_local: frame_real}."""
    import shutil
    if os.path.exists(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(carpeta)
    mapa = {}
    cap = cv2.VideoCapture(ruta_video)
    for local, fi in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=escala, fy=escala)
        cv2.imwrite(os.path.join(carpeta, f"{local:04d}.jpg"), small)
        mapa[local] = fi
    cap.release()
    return mapa


FPS_DEF = 29.97002997002997


def posicion_solar_evento(cfg, t_offset_s=None, verbose=True):
    """Sol en el instante de la tronadura. 0 = Norte, horario."""
    import datetime
    from zoneinfo import ZoneInfo

    t0 = datetime.datetime.fromisoformat(cfg.ts_inicio_video).replace(
        tzinfo=ZoneInfo(cfg.tz_local))
    if t_offset_s is None:
        fps = FPS_DEF
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(cfg.video)
            fps = cap.get(5) or FPS_DEF
            cap.release()
        except Exception:
            pass
        t_offset_s = (cfg.pivot_frame or 0) / fps
    dt = t0 + datetime.timedelta(seconds=float(t_offset_s))
    lat, lon = cfg.frame_data["lat_deg"], cfg.frame_data["lon_deg"]

    try:
        from pysolar.solar import get_altitude, get_azimuth
        el, az = get_altitude(lat, lon, dt), get_azimuth(lat, lon, dt)
        fuente = "pysolar"
    except Exception:
        az, el = _sol_noaa(dt.astimezone(datetime.timezone.utc), lat, lon)
        fuente = "NOAA compacto"

    if el <= 0:
        raise ValueError(
            f"Sol bajo el horizonte (elevacion {el:.1f} deg) para "
            f"{dt:%Y-%m-%d %H:%M} {cfg.tz_local}. Revisa ts_inicio_video.")
    a, e = math.radians(az), math.radians(el)
    vec = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    if verbose:
        print(f"  sol ({fuente}) en {dt:%Y-%m-%d %H:%M:%S}: azimut {az:.1f} deg, "
              f"elevacion {el:.1f} deg")
        print(f"    la sombra mide {1/math.tan(e):.2f} x la altura")
    return dict(azimut_deg=float(az), elevacion_deg=float(el),
                vec_sol=[float(x) for x in vec], fuente=fuente,
                instante=dt.isoformat())


def _sol_noaa(dt_utc, lat, lon):
    import datetime
    d = (dt_utc.replace(tzinfo=None) - datetime.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    h = math.radians((gmst * 15 + lon) % 360) - ra
    la = math.radians(lat)
    el = math.asin(math.sin(la) * math.sin(dec) +
                   math.cos(la) * math.cos(dec) * math.cos(h))
    az = math.atan2(-math.cos(dec) * math.sin(h),
                    math.sin(dec) * math.cos(la) - math.cos(dec) * math.sin(la) * math.cos(h))
    return math.degrees(az) % 360, math.degrees(el)


def pivote_px(cfg, escala=None):
    """Pixel del pivote en resolucion de TRABAJO."""
    escala = cfg.escala if escala is None else float(escala)
    try:
        with open(cfg.art_geometria, "r", encoding="utf-8") as fh:
            g = json.load(fh)
        px_n, py_n = (float(v) for v in g["pivote_px_nat"])
        return px_n * escala, py_n * escala
    except (OSError, KeyError, TypeError, ValueError):
        pass
    c = normalizar_calibracion(leer_calibracion(cfg.calibracion))
    return c["pivot_x_nat"] * escala, c["pivot_y_nat"] * escala


class CamaraGeo:
    """Camara pinhole con pose en UTM. Todo lo de mundo cuelga de aqui."""

    def __init__(self, K, C_utm, R_c_w, W, H, escala=0.5, z_piso=None):
        self.K = np.asarray(K, float)
        self.C = np.asarray(C_utm, float)
        self.R = np.asarray(R_c_w, float)
        self.W, self.H = int(W), int(H)
        self.escala = float(escala)
        self.z_piso = float(z_piso) if z_piso is not None else None
        self.origen_utm = None
        self.sol = None
        self._Ki = np.linalg.inv(self.K)

    @classmethod
    def desde_json(cls, ruta, escala=0.5, z_piso=None):
        with open(ruta, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        origen = np.array(d["origen_utm"], float)
        C = origen + np.array(d["posicion_blender"], float)
        f = float(d["focal_px"]); W = int(d["ancho_px"]); H = int(d["alto_px"])
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        cam = cls(K, C, np.array(d["R_c_w"], float), W, H, escala,
                  z_piso if z_piso is not None else origen[2])
        cam.origen_utm = origen
        cam.sol = d.get("sol")
        return cam

    @classmethod
    def desde_config(cls, cfg, dem=None, escala=None, resolver_pivote=True,
                     verbose=True):
        """Camara a partir de `Config` — o sea, de la telemetria del MP4."""
        from pyproj import Transformer

        fd = dict(cfg.frame_data)
        escala = cfg.escala if escala is None else float(escala)
        W = int(round(fd["width"] * escala))
        H = int(round(fd["height"] * escala))
        f = float(fd["focal"]) * escala
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])

        crs = str(dem["crs"]) if dem is not None else "EPSG:32719"
        E, N = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            fd["lon_deg"], fd["lat_deg"])
        C = np.array([E, N, float(fd["absolute_altitude_m"])])
        R = cls.rotacion(fd["drone_yaw_deg"], fd["gimbal_pitch_deg"])

        cam = cls(K, C, R, W, H, escala)
        try:
            cam.sol = posicion_solar_evento(cfg, verbose=False)
        except Exception:
            cam.sol = None

        if dem is not None and resolver_pivote:
            pu, pv = pivote_px(cfg, escala)
            P = raycast_pixeles(cam, dem, pu, pv)[0]
            if not np.isfinite(P).all():
                raise RuntimeError(
                    "El rayo del pivote no intersecta el DEM. Revisa que el "
                    "DEM cubra la zona del vuelo y que la pose de la "
                    "telemetria sea la correcta.")
            cam.origen_utm = P
            cam.z_piso = float(P[2])
        else:
            cam.origen_utm = None

        if verbose:
            cam.resumen()
        return cam

    @staticmethod
    def rotacion(yaw_deg, pitch_deg):
        """R_c_w a partir de yaw del dron y pitch del gimbal."""
        def Ry(d):
            a = math.radians(d); c, s = math.cos(a), math.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def Rz(d):
            a = math.radians(d); c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        ENU = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
        R_c_g = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
        return ENU @ Rz(yaw_deg) @ Ry(pitch_deg) @ R_c_g

    def rayos(self, u, v, R=None):
        """Pixeles de TRABAJO -> direcciones unitarias en mundo. (N,3)"""
        u = np.atleast_1d(np.asarray(u, float)).ravel()
        v = np.atleast_1d(np.asarray(v, float)).ravel()
        pix = np.stack([u, v, np.ones_like(u)], axis=1)
        d = (pix @ self._Ki.T) @ (self.R if R is None else R).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def a_plano(self, u, v, z):
        """Pixel -> punto UTM sobre el plano horizontal z. (N,3), NaN si no corta."""
        d = self.rayos(u, v)
        z = np.broadcast_to(np.atleast_1d(np.asarray(z, float)), (len(d),))
        t = (z - self.C[2]) / d[:, 2]
        P = self.C[None, :] + t[:, None] * d
        P[t <= 0] = np.nan
        return P

    def a_pixel(self, P_utm):
        """UTM -> pixel de trabajo. Devuelve (uv (N,2), delante (N,))."""
        P = np.atleast_2d(np.asarray(P_utm, float))
        Pc = (P - self.C) @ self.R
        delante = Pc[:, 2] > 1e-9
        zc = np.where(delante, Pc[:, 2], np.nan)
        uv = np.stack([self.K[0, 0] * Pc[:, 0] / zc + self.K[0, 2],
                       self.K[1, 1] * Pc[:, 1] / zc + self.K[1, 2]], axis=1)
        return uv, delante

    def escala_local(self, u, v, z, eps=2.0):
        """m por pixel NATIVO en (u,v) sobre el plano z, en u y en v."""
        P0 = self.a_plano(u, v, z)
        Pu = self.a_plano(np.asarray(u, float) + eps, v, z)
        Pv = self.a_plano(u, np.asarray(v, float) + eps, z)
        su = np.linalg.norm((Pu - P0)[:, :2], axis=1) / eps * self.escala
        sv = np.linalg.norm((Pv - P0)[:, :2], axis=1) / eps * self.escala
        return su, sv

    def resumen(self):
        def az(w):
            return math.degrees(math.atan2(w[0], w[1])) % 360
        d = self.R[:, 2]
        print(f"  camara UTM : E {self.C[0]:.1f}  N {self.C[1]:.1f}  z {self.C[2]:.1f}")
        print(f"  mirada     : azimut {az(d):.1f} deg | depresion "
              f"{math.degrees(math.asin(-d[2])):.1f} deg")
        print(f"  u_img      -> azimut {az(self.R[:, 0]):.1f} deg")
        print(f"  v_img      -> azimut {az(self.R[:, 1]):.1f} deg")
        if self.z_piso is not None:
            u = np.full(3, self.W / 2.0)
            v = np.array([1.0, self.H / 2.0, self.H - 1.0])
            su, _ = self.escala_local(u, v, self.z_piso)
            print(f"  escala en el plano del piso (m/px nativo): "
                  f"arriba {su[0]:.4f} | centro {su[1]:.4f} | abajo {su[2]:.4f}"
                  f"  -> factor {su[0]/su[2]:.2f}")


def cargar_dem(ruta_dem, bounds=None, verbose=True):
    """DEM -> dict con el interpolador bilineal, los ejes y el transform."""
    import rasterio
    from rasterio.windows import from_bounds
    from scipy.interpolate import RegularGridInterpolator

    with rasterio.open(ruta_dem) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(f"DEM debe estar en CRS proyectado (UTM), no {src.crs}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError("DEM rotado: no soportado")
        if bounds is None:
            Z = src.read(1).astype(float)
            T = src.transform
        else:
            ven = from_bounds(*bounds, transform=src.transform)
            ven = ven.round_offsets().round_lengths()
            Z = src.read(1, window=ven).astype(float)
            T = src.window_transform(ven)
        if src.nodata is not None:
            Z[Z == src.nodata] = np.nan
        crs, forma_archivo = src.crs, src.shape

    ny, nx = Z.shape
    xs = T.c + (np.arange(nx) + 0.5) * T.a
    ys_desc = T.f + (np.arange(ny) + 0.5) * T.e
    dem = np.flipud(Z)
    ys = ys_desc[::-1]

    Z_interp = RegularGridInterpolator((ys, xs), dem, method="linear",
                                       bounds_error=False, fill_value=np.nan)

    if verbose:
        print(f"  DEM {crs} | archivo {forma_archivo} | ventana {Z.shape} | "
              f"{abs(T.a):.1f} m | z {np.nanmin(Z):.0f}-{np.nanmax(Z):.0f} m")

    return dict(
        Z=Z, dem=dem, xs=xs, ys=ys, transform=T, crs=crs, res=float(abs(T.a)),
        bounds=bounds, Z_interp=Z_interp,
        z=lambda x, y: Z_interp(np.stack([np.atleast_1d(y),
                                          np.atleast_1d(x)], -1)))

def _marchar(cam, dem, d, t_max, paso, n_bisec):
    """Marchado + biseccion de un haz de rayos contra el DEM. Devuelve (N,3)."""
    C = cam.C
    n = len(d)
    t = np.full(n, np.nan)
    tc = np.full(n, paso)
    P = C + tc[:, None] * d
    fprev = P[:, 2] - dem["z"](P[:, 0], P[:, 1])
    for _ in range(int(t_max / paso)):
        tc = tc + paso
        P = C + tc[:, None] * d
        fv = P[:, 2] - dem["z"](P[:, 0], P[:, 1])
        cr = np.isnan(t) & np.isfinite(fv) & np.isfinite(fprev) & (fv <= 0) & (fprev > 0)
        if cr.any():
            lo = tc[cr] - paso; hi = tc[cr].copy()
            for _ in range(n_bisec):
                mid = (lo + hi) / 2
                Pm = C + mid[:, None] * d[cr]
                pos = (Pm[:, 2] - dem["z"](Pm[:, 0], Pm[:, 1])) > 0
                lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
            t[cr] = (lo + hi) / 2
        fprev = np.where(np.isnan(t), fv, fprev)
        if np.isfinite(t).all():
            break
    return C + t[:, None] * d


def raycast_pixeles(cam, dem, u, v, R=None, t_max=2500.0, paso=8.0, n_bisec=24):
    """Interseccion rayo-DEM para pixeles sueltos. Devuelve (N,3) en UTM."""
    u = np.atleast_1d(np.asarray(u, float)).ravel()
    v = np.atleast_1d(np.asarray(v, float)).ravel()
    return _marchar(cam, dem, cam.rayos(u, v, R=R), t_max, paso, n_bisec)


def raycast_dem(cam, dem, step=4, R=None, t_max=2500.0, paso=8.0, n_bisec=24):
    """Un rayo por pixel contra el DEM. Devuelve XM, YM, ZM (NaN si no corta)."""
    uu, vv = np.meshgrid(np.arange(0, cam.W, step, float),
                         np.arange(0, cam.H, step, float))
    forma = uu.shape
    P = _marchar(cam, dem, cam.rayos(uu.ravel(), vv.ravel(), R=R),
                 t_max, paso, n_bisec)
    return (P[:, 0].reshape(forma), P[:, 1].reshape(forma), P[:, 2].reshape(forma))


def verificar_escala(cam, gsd_nativo_airblast):
    """Contrasta el rango camara-pivote del modelo contra el que implica el airblast."""
    if cam.origen_utm is None:
        print("  (sin origen_utm: no se puede verificar)"); return None
    d_mod = float(np.linalg.norm(cam.C - cam.origen_utm))
    d_ab = float(gsd_nativo_airblast / cam.escala * cam.K[0, 0])
    err = 100 * (d_mod / d_ab - 1)
    print(f"  rango camara->pivote (modelo camara + DEM) : {d_mod:7.1f} m")
    print(f"  rango implicado por el airblast (PPM_NAT)  : {d_ab:7.1f} m")
    print(f"  discrepancia : {d_mod-d_ab:+.1f} m  ({err:+.1f} %)")
    dz = 102.0
    k = abs(cam.C[2] - cam.origen_utm[2]) / d_mod
    print(f"  a modo de contraste, un desfase real de {dz:.0f} m en el DEM daria "
          f"{100*((d_mod-dz*k)/d_ab-1):+.0f} % / {100*((d_mod+dz*k)/d_ab-1):+.0f} %")
    return dict(d_modelo=d_mod, d_airblast=d_ab, error_pct=err)


def barrido_yaw(cam, dem, ruta_ortofoto, bg_gris, yaw0, pitch, delta=20.0,
                paso=2.5, step=8, margen=22, verbose=True):
    """Valida la ORIENTACION reproyectando la ortofoto satelital al cuadro."""
    import rasterio
    sat = rasterio.open(ruta_ortofoto)

    def prep(a, s1=2, s2=14):
        a = np.nan_to_num(np.asarray(a, np.float32),
                          nan=float(np.nanmean(a))).astype(np.float32)
        return cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)

    obs = np.asarray(bg_gris, float)[::step, ::step]
    yaws = np.arange(yaw0 - delta, yaw0 + delta + 1e-6, paso)
    cor = []
    for y in yaws:
        X, Y, _ = raycast_dem(cam, dem, step=step, R=CamaraGeo.rotacion(y, pitch))
        m = np.isfinite(X)
        S = np.full(X.shape, np.nan)
        if m.any():
            r, c = rasterio.transform.rowcol(sat.transform, X[m], Y[m])
            r = np.clip(np.asarray(r), 0, sat.height - 1)
            c = np.clip(np.asarray(c), 0, sat.width - 1)
            w = rasterio.windows.Window(int(c.min()), int(r.min()),
                                        int(c.max() - c.min() + 1),
                                        int(r.max() - r.min() + 1))
            band = sat.read(1, window=w).astype(float)
            S[m] = band[r - int(r.min()), c - int(c.min())]
        o = obs[:X.shape[0], :X.shape[1]]
        plant = prep(o)[margen:-margen, margen:-margen]
        rr = cv2.matchTemplate(prep(S), plant.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        cor.append(float(cv2.minMaxLoc(rr)[1]))
        if verbose:
            print(f"    yaw {y:7.1f}  r = {cor[-1]:.3f}", end="\r")
    cor = np.array(cor)
    best = float(yaws[np.argmax(cor)])
    if verbose:
        anchos = yaws[cor >= cor.max() - 0.05 * (cor.max() - cor.min() + 1e-9) * 0]
        print(f"\n  yaw de metadatos {yaw0:7.1f} deg")
        print(f"  yaw que maximiza la correlacion {best:7.1f} deg  "
              f"(r = {cor.max():.3f})")
        print(f"  diferencia {best-yaw0:+.1f} deg  ->  el mismo sesgo se traslada "
              f"al azimut del viento")
    return yaws, cor, best


def georreferenciar_frame(cfg, cam, dem, puntos=None, ruta_csv=None,
                          verbose=True) -> dict:
    """Puntos de control del cuadro -> UTM, lat/lon y rango. Escribe el CSV."""
    from pyproj import Transformer

    W, H = cam.W, cam.H
    if puntos is None:
        pu, pv = pivote_px(cfg, cam.escala)
        puntos = [("pivote",      pu,       pv),
                  ("esq_sup_izq", 0.5,      0.5),
                  ("esq_sup_der", W - 0.5,  0.5),
                  ("esq_inf_der", W - 0.5,  H - 0.5),
                  ("esq_inf_izq", 0.5,      H - 0.5),
                  ("centro",      W / 2.0,  H / 2.0)]

    us = np.array([p[1] for p in puntos], float)
    vs = np.array([p[2] for p in puntos], float)
    pts = raycast_pixeles(cam, dem, us, vs)
    ok = np.isfinite(pts).all(axis=1)
    rango = np.where(ok, np.linalg.norm(pts - cam.C, axis=1), np.nan)

    tr = Transformer.from_crs(str(dem["crs"]), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pts[:, 0], pts[:, 1])

    filas = []
    if verbose:
        print(f"{'punto':<14}{'px_4K':>16}{'UTM E':>12}{'UTM N':>13}{'z':>9}"
              f"{'lat':>12}{'lon':>12}{'dist':>8}")
    for i, (nom, u, v) in enumerate(puntos):
        u4k, v4k = u / cam.escala, v / cam.escala
        if not ok[i]:
            if verbose:
                print(f"{nom:<14}{f'{u4k:.0f},{v4k:.0f}':>16}"
                      f"   <- el rayo NO corta el DEM")
            filas.append([nom, f"{u4k:.1f}", f"{v4k:.1f}", f"{u:.1f}", f"{v:.1f}",
                          "", "", "", "", ""])
            continue
        if verbose:
            print(f"{nom:<14}{f'{u4k:.0f},{v4k:.0f}':>16}{pts[i,0]:>12.2f}"
                  f"{pts[i,1]:>13.2f}{pts[i,2]:>9.2f}{lat[i]:>12.6f}"
                  f"{lon[i]:>12.6f}{rango[i]:>8.0f}")
        filas.append([nom, f"{u4k:.1f}", f"{v4k:.1f}", f"{u:.1f}", f"{v:.1f}",
                      f"{pts[i,0]:.2f}", f"{pts[i,1]:.2f}", f"{pts[i,2]:.2f}",
                      f"{lat[i]:.7f}", f"{lon[i]:.7f}"])

    ruta_csv = ruta_csv or cfg.ruta(0, "georreferencia_frame.csv")
    guardar_csv(filas,
                ["punto", "px_x_4K", "px_y_4K", "px_x_trab", "px_y_trab",
                 "utm_e", "utm_n", "z_m", "lat", "lon"], ruta_csv)
    if verbose:
        print(f"\nCRS: {dem['crs']} (UTM 19S) | frame de referencia: "
              f"f{cfg.pivot_frame}")
    return dict(puntos=puntos, pts=pts, ok=ok, lat=np.asarray(lat),
                lon=np.asarray(lon), rango=rango, csv=ruta_csv)


_CARDINALES = (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("O", (-1, 0)))


def _sombra_horizontal(sol):
    """Direccion horizontal de la sombra (unitaria), desde el vector solar."""
    vs = np.asarray(sol["vec_sol"], float)
    dh = -vs.copy()
    dh[2] = 0.0
    return dh / np.linalg.norm(dh)


def _fondo_planta(ax, cfg, dem, e0, e1, n0, n1, ruta_ortofoto=None) -> bool:
    """Textura del terreno bajo la vista en planta. True si uso la ortofoto."""
    ruta_ortofoto = ruta_ortofoto or (cfg.satelital if cfg is not None else None)
    if ruta_ortofoto and os.path.exists(ruta_ortofoto):
        try:
            import rasterio
            from rasterio.windows import from_bounds
            with rasterio.open(ruta_ortofoto) as sat:
                w = from_bounds(e0, n0, e1, n1, sat.transform)
                nb = min(3, sat.count)
                arr = sat.read(list(range(1, nb + 1)), window=w,
                               out_shape=(nb, 900, 900), boundless=True,
                               fill_value=0)
                img = np.moveaxis(arr, 0, -1)
                if img.dtype != np.uint8:
                    img = (255 * (img - img.min())
                           / max(np.ptp(img), 1)).astype(np.uint8)
                ax.imshow(img.squeeze(), extent=[e0, e1, n0, n1], origin="upper",
                          cmap=None if nb == 3 else "gray", zorder=0)
            return True
        except Exception as ex:
            print(f"  [aviso] no pude leer la ortofoto ({ex}); uso sombreado")

    xs, ys, Z = dem["xs"], dem["ys"], dem["dem"]
    ci = slice(int(np.searchsorted(xs, e0)), int(np.searchsorted(xs, e1)))
    ri = slice(int(np.searchsorted(ys, n0)), int(np.searchsorted(ys, n1)))
    Zc = Z[ri, ci]
    if Zc.size < 4:
        return False
    dn, de = np.gradient(Zc, dem["res"])
    pend = np.arctan(np.hypot(de, dn))
    asp = np.arctan2(-de, dn)
    az, el = np.radians(315.0), np.radians(45.0)
    hs = (np.sin(el) * np.cos(pend)
          + np.cos(el) * np.sin(pend) * np.cos(az - asp))
    ax.imshow(hs, cmap="gray", origin="lower", zorder=0,
              extent=[xs[ci][0], xs[ci][-1], ys[ri][0], ys[ri][-1]])
    return False


def _muestrear_ortofoto(ruta, X, Y, verbose=True):
    """Color de la ortofoto en cada punto UTM. Devuelve (ny, nx, 3) uint8."""
    m = np.isfinite(X) & np.isfinite(Y)
    if not m.any():
        return None
    try:
        import rasterio
        with rasterio.open(ruta) as sat:
            r, c = rasterio.transform.rowcol(sat.transform, X[m], Y[m])
            r = np.clip(np.asarray(r).ravel(), 0, sat.height - 1)
            c = np.clip(np.asarray(c).ravel(), 0, sat.width - 1)
            r0, c0 = int(r.min()), int(c.min())
            w = rasterio.windows.Window(c0, r0, int(c.max() - c0 + 1),
                                        int(r.max() - r0 + 1))
            nb = min(3, sat.count)
            banda = sat.read(list(range(1, nb + 1)), window=w)
    except Exception as ex:
        if verbose:
            print(f"  [aviso] no pude muestrear la ortofoto ({ex})")
        return None
    if banda.dtype != np.uint8:
        banda = (255.0 * (banda - banda.min())
                 / max(np.ptp(banda), 1)).astype(np.uint8)
    vals = banda[:, r - r0, c - c0]
    if vals.shape[0] == 1:
        vals = np.repeat(vals, 3, axis=0)
    out = np.zeros(X.shape + (3,), np.uint8)
    out[m] = np.moveaxis(vals, 0, -1)
    return out


def vista_sintetica(cfg, cam, dem, step=6, t_max=2500.0, ruta_ortofoto=None,
                    verbose=True) -> dict:
    """El cuadro que veria el dron si la escena fuera SOLO el DEM y la ortofoto."""
    X, Y, Z = raycast_dem(cam, dem, step=step, t_max=t_max)
    u = np.arange(0, cam.W, step, dtype=float)
    v = np.arange(0, cam.H, step, dtype=float)
    m = np.isfinite(Z)

    if verbose:
        print(f"  vista sintetica: {Z.size} rayos (step {step}) | "
              f"{100 * m.mean():.0f}% corta el DEM | "
              f"cota {np.nanmin(Z):.0f}-{np.nanmax(Z):.0f} m")
        if m.mean() < 0.5:
            print("   AVISO: mas de la mitad del cuadro no corta el DEM. O el "
                  "dron mira por encima del horizonte, o el DEM no cubre lo "
                  "que se ve.")

    ruta_ortofoto = ruta_ortofoto or (cfg.satelital if cfg is not None else None)
    rgb = None
    if ruta_ortofoto and os.path.exists(ruta_ortofoto):
        rgb = _muestrear_ortofoto(ruta_ortofoto, X, Y, verbose=verbose)
    elif verbose:
        print(f"  (sin ortofoto en {ruta_corta(ruta_ortofoto)}: la vista sintetica sale "
              f"coloreada por cota)")
    return dict(rgb=rgb, X=X, Y=Y, Z=Z, u=u, v=v, step=step,
                cobertura=float(m.mean()))


def graficar_vista_sintetica(cfg, cam=None, dem=None, bg=None, step=6,
                             ruta_png=None, verbose=True):
    """Dos paneles en el MISMO plano imagen y a la MISMA escala."""
    import matplotlib.pyplot as plt

    if dem is None:
        dem = cargar_dem(cfg.dem, verbose=verbose)
    if cam is None:
        cam = CamaraGeo.desde_config(cfg, dem=dem, verbose=verbose)
    W, H = cam.W, cam.H

    VS = vista_sintetica(cfg, cam, dem, step=step, verbose=verbose)

    if bg is None:
        r = cfg.ruta(0, "bg_color.png")
        bg = cv2.imread(r) if os.path.exists(r) else None
    if bg is None:
        with LectorVideo(cfg.video, cfg.escala) as lec:
            bg = lec[int(cfg.pivot_frame)]
    real = bg[..., ::-1] if np.ndim(bg) == 3 else bg

    ancho_panel = 7.6
    fig, ax = plt.subplots(1, 2, figsize=(2 * ancho_panel,
                                          ancho_panel * H / W + 1.4))
    EXT = [0, W, H, 0]

    a = ax[0]
    a.imshow(np.asarray(real), extent=EXT, aspect="equal",
             cmap=None if np.ndim(bg) == 3 else "gray")
    a.set_title(f"Frame real (f{cfg.pivot_frame})", fontsize=10)

    b = ax[1]
    if VS["rgb"] is not None:
        b.imshow(VS["rgb"], extent=EXT, aspect="equal", interpolation="bilinear")
        b.set_title("Sintética (DEM)", fontsize=10)
    else:
        h = b.imshow(VS["Z"], cmap="terrain", extent=EXT, aspect="equal",
                     interpolation="bilinear")
        plt.colorbar(h, ax=b, fraction=0.035, pad=0.02, label="cota [m]")
        b.set_title("Sintética (DEM)", fontsize=10)

    for a_ in ax:
        a_.set_xlim(0, W); a_.set_ylim(H, 0)
        a_.set_xticks([]); a_.set_yticks([])

    fig.suptitle(f"Telemetría + "
                 f"{os.path.basename(cfg.dem_nombre)} "
                 f"({dem['res']:.1f} m) + {os.path.basename(cfg.satelital_nombre)}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if ruta_png:
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
        plt.savefig(ruta_png, dpi=140, bbox_inches="tight")
        print(f"[fig] {ruta_corta(ruta_png)}")
    VS["fig"] = fig
    VS["dem"] = dem
    VS["cam"] = cam
    return VS


def brujula_en_imagen(ax, cam, u0, v0, z, largo_m=40.0, color="#FFEB3B",
                      dem=None, sol=None, fontsize=11, marcar_origen=False):
    """Puntos cardinales PROYECTADOS sobre el cuadro."""
    P0 = cam.a_plano(u0, v0, z)[0]
    if dem is not None:
        P0[2] = float(dem["z"](P0[0], P0[1])[0])

    def punta(de, dn):
        P1 = P0 + np.array([de * largo_m, dn * largo_m, 0.0])
        if dem is not None:
            P1[2] = float(dem["z"](P1[0], P1[1])[0])
        return P1

    for et, (de, dn) in _CARDINALES:
        uv, _ = cam.a_pixel(np.stack([P0, punta(de, dn)]))
        pr = et in ("N", "E")
        ax.annotate("", xy=(uv[1, 0], uv[1, 1]), xytext=(uv[0, 0], uv[0, 1]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6 if pr else 1.2,
                                    color=color, alpha=1.0 if pr else .55),
                    annotation_clip=True, zorder=10)
        ax.text(np.clip(uv[1, 0], 18, cam.W - 18),
                np.clip(uv[1, 1], 18, cam.H - 18), et, color=color,
                fontsize=fontsize + 2 if pr else fontsize,
                fontweight="bold" if pr else "normal", zorder=11,
                ha="center", va="center")
    if sol is not None:
        dh = _sombra_horizontal(sol)
        uv, _ = cam.a_pixel(np.stack([P0, punta(dh[0], dh[1])]))
        ax.annotate("", xy=(uv[1, 0], uv[1, 1]), xytext=(uv[0, 0], uv[0, 1]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#FF5252"),
                    zorder=10)
        ax.text(np.clip(uv[1, 0], 40, cam.W - 40),
                np.clip(uv[1, 1], 24, cam.H - 24), "sombra",
                color="#FF5252", fontsize=fontsize - 2, fontweight="bold", zorder=11)
    if marcar_origen:
        ax.plot(u0, v0, "o", color=color, ms=5, zorder=11)
    return ax


def panel_cuadro_cardinales(ax, cam, dem, bg, largo_m=36.0, ancla=(0.62, 0.54),
                            titulo=None):
    """Panel A: el frame del video con la brujula apoyada en el terreno real."""
    u0, v0 = ancla[0] * cam.W, ancla[1] * cam.H
    im = bg[..., ::-1] if np.ndim(bg) == 3 else bg
    ax.imshow((np.asarray(im) * 0.74).astype(np.uint8),
              cmap=None if np.ndim(bg) == 3 else "gray")
    brujula_en_imagen(ax, cam, u0, v0, cam.z_piso, largo_m=largo_m, dem=dem,
                      sol=getattr(cam, "sol", None))
    ax.set_xlim(0, cam.W); ax.set_ylim(cam.H, 0)
    ax.set_xlabel("u (px de trabajo)"); ax.set_ylabel("v (px de trabajo)")
    ax.set_title(titulo or "Cuadro del video con brújula proyectada y sombra "
                 "predicha", fontsize=9.5)
    return u0, v0


def panel_planta_cardinales(ax, cam, dem, XM, YM, largo_m=60.0, margen_m=190.0,
                            ancla=(0.62, 0.54), cfg=None, ruta_ortofoto=None,
                            titulo=None):
    """Panel B: vista en planta con la huella del cuadro y las MISMAS flechas."""
    e0, e1 = np.nanmin(XM) - margen_m, np.nanmax(XM) + margen_m
    n0, n1 = np.nanmin(YM) - margen_m, np.nanmax(YM) + margen_m
    _fondo_planta(ax, cfg, dem, e0, e1, n0, n1, ruta_ortofoto=ruta_ortofoto)

    hu = np.concatenate([XM[0, :], XM[:, -1], XM[-1, ::-1], XM[::-1, 0]])
    hn = np.concatenate([YM[0, :], YM[:, -1], YM[-1, ::-1], YM[::-1, 0]])
    ax.fill(hu, hn, color="#00E5FF", alpha=.10, zorder=2)
    ax.plot(hu, hn, color="#00E5FF", lw=1.8, zorder=3, label="huella del cuadro")
    ax.plot(cam.C[0], cam.C[1], "o", color="#D50000", ms=10, zorder=6,
            label="dron")

    P0 = cam.a_plano(ancla[0] * cam.W, ancla[1] * cam.H, cam.z_piso)[0]
    P0[2] = float(dem["z"](P0[0], P0[1])[0])
    for et, (de, dn) in _CARDINALES:
        pr = et in ("N", "E")
        ax.annotate("", xy=(P0[0] + de * largo_m, P0[1] + dn * largo_m),
                    xytext=(P0[0], P0[1]), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", lw=2.6 if pr else 1.2,
                                    color="#FFD600" if pr else "#FFF176"))
        ax.text(P0[0] + de * largo_m * 1.3, P0[1] + dn * largo_m * 1.3, et,
                color="#FFD600", fontsize=13 if pr else 10,
                fontweight="bold" if pr else "normal", ha="center", va="center",
                zorder=10)
    if getattr(cam, "sol", None):
        dh = _sombra_horizontal(cam.sol)
        ax.annotate("", xy=(P0[0] + dh[0] * largo_m, P0[1] + dh[1] * largo_m),
                    xytext=(P0[0], P0[1]), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#FF5252"))

    ax.set_xlim(e0, e1); ax.set_ylim(n0, n1); ax.set_aspect("equal")
    ax.set_xlabel("Este UTM (m)"); ax.set_ylabel("Norte UTM (m)")
    ax.set_title(titulo or "Vista en planta", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=.85)
    ax.annotate("N", xy=(.945, .965), xytext=(.945, .855),
                xycoords="axes fraction", ha="center", va="bottom",
                fontsize=11, fontweight="bold", zorder=11,
                arrowprops=dict(arrowstyle="-|>", lw=2, color="k"),
                bbox=dict(fc="w", ec="none", alpha=.75, boxstyle="circle,pad=0.18"))
    return P0, (e0, e1, n0, n1)


def graficar_puntos_cardinales(cfg, cam, dem, bg=None, XM=None, YM=None,
                               largo_m=60.0, ruta_png=None, verbose=True):
    """Los puntos cardinales verificados contra el DEM, en dos paneles."""
    import matplotlib.pyplot as plt

    if XM is None or YM is None:
        XM, YM, _ = raycast_dem(cam, dem, step=6)
    if bg is None:
        r = cfg.ruta(0, "bg_color.png")
        bg = cv2.imread(r) if os.path.exists(r) else None
    if bg is None:
        with LectorVideo(cfg.video, cfg.escala) as lec:
            bg = lec[int(cfg.pivot_frame)]

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.6),
                           gridspec_kw=dict(width_ratios=[1.30, 1.0]))
    fig.suptitle("Puntos cardinales verificados contra el DEM de la mina",
                 fontsize=13, fontweight="bold")
    panel_cuadro_cardinales(ax[0], cam, dem, bg, largo_m=largo_m * 0.6)
    panel_planta_cardinales(ax[1], cam, dem, XM, YM, largo_m=largo_m, cfg=cfg)

    plt.tight_layout()
    if ruta_png:
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
        plt.savefig(ruta_png, dpi=140, bbox_inches="tight")
        print(f"[fig] {ruta_corta(ruta_png)}")
    return fig


_CAMPOS = {
    ".3.3.4.1.2": ("lat_deg", math.degrees, False),
    ".3.3.4.1.3": ("lon_deg", math.degrees, False),
    ".3.3.4.2":   ("absolute_altitude_m", lambda v: v / 1000.0, False),
    ".3.3.3.3":   ("drone_yaw_deg", lambda v: v / 10.0, True),
    ".3.3.3.1":   ("drone_pitch_deg", lambda v: v / 10.0, True),
    ".3.3.3.2":   ("drone_roll_deg", lambda v: v / 10.0, True),
    ".3.4.3.1":   ("gimbal_pitch_deg", lambda v: v / 10.0, True),
    ".3.4.3.3":   ("gimbal_yaw_deg", lambda v: v / 10.0, True),
    ".2.2.1":     ("width", int, False),
    ".2.2.2":     ("height", int, False),
    ".2.2.3":     ("fps", float, False),
}


def _boxes(fh, ini, fin):
    fh.seek(ini)
    while fh.tell() < fin - 8:
        off = fh.tell()
        cab = fh.read(8)
        if len(cab) < 8:
            return
        tam, tipo = struct.unpack(">I4s", cab)
        cuerpo = off + 8
        if tam == 1:
            tam = struct.unpack(">Q", fh.read(8))[0]
            cuerpo = off + 16
        elif tam == 0:
            tam = fin - off
        if tam < 8:
            return
        yield tipo.decode("latin1"), cuerpo, off + tam
        fh.seek(off + tam)


def _buscar(fh, ini, fin, camino):
    """Devuelve (inicio_cuerpo, fin) del primer box que siga `camino`."""
    if not camino:
        return ini, fin
    for tipo, c0, c1 in _boxes(fh, ini, fin):
        if tipo == camino[0]:
            r = _buscar(fh, c0, c1, camino[1:])
            if r:
                return r
    return None


def _tabla(fh, ini, fin, tipo):
    r = _buscar(fh, ini, fin, [tipo])
    if not r:
        return None
    fh.seek(r[0])
    return fh.read(r[1] - r[0])


def muestras_djmd(ruta_video, codec=b"djmd"):
    """Offsets y tamanos de las muestras de la pista de datos DJI."""
    with open(ruta_video, "rb") as fh:
        fin = os.path.getsize(ruta_video)
        moov = _buscar(fh, 0, fin, ["moov"])
        if not moov:
            raise ValueError("MP4 sin box 'moov'")
        for tipo, t0, t1 in _boxes(fh, *moov):
            if tipo != "trak":
                continue
            stbl = _buscar(fh, t0, t1, ["mdia", "minf", "stbl"])
            if not stbl:
                continue
            stsd = _tabla(fh, *stbl, "stsd")
            if not stsd or codec not in stsd:
                continue

            stsz = _tabla(fh, *stbl, "stsz")
            uniforme = struct.unpack_from(">I", stsz, 4)[0]
            cnt = struct.unpack_from(">I", stsz, 8)[0]
            tam = ([uniforme] * cnt if uniforme else
                   list(struct.unpack_from(f">{cnt}I", stsz, 12)))

            co = _tabla(fh, *stbl, "stco")
            ancho = 4
            if co is None:
                co = _tabla(fh, *stbl, "co64"); ancho = 8
            nch = struct.unpack_from(">I", co, 4)[0]
            chunks = list(struct.unpack_from(
                f">{nch}{'Q' if ancho == 8 else 'I'}", co, 8))

            stsc = _tabla(fh, *stbl, "stsc")
            ne = struct.unpack_from(">I", stsc, 4)[0]
            ent = [struct.unpack_from(">3I", stsc, 8 + 12 * i) for i in range(ne)]

            offs, k = [], 0
            for i, base in enumerate(chunks, start=1):
                por_chunk = ent[0][1]
                for j in range(len(ent)):
                    if ent[j][0] <= i:
                        por_chunk = ent[j][1]
                p = base
                for _ in range(por_chunk):
                    if k >= cnt:
                        break
                    offs.append((p, tam[k])); p += tam[k]; k += 1
            return offs
    raise ValueError(f"El MP4 no tiene pista de datos {codec.decode()}. "
                     f"Ese modelo de dron no embebe telemetria, o el archivo "
                     f"fue recodificado y la perdio.")


def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i


def _parece_mensaje(s):
    try:
        i = n = 0
        while i < len(s) and n < 48:
            k, i = _varint(s, i)
            fn, wt = k >> 3, k & 7
            if fn == 0 or wt not in (0, 1, 2, 5):
                return False
            if wt == 0:
                _, i = _varint(s, i)
            elif wt == 1:
                i += 8
            elif wt == 5:
                i += 4
            else:
                l, i = _varint(s, i); i += l
            n += 1
        return i == len(s)
    except Exception:
        return False


def _recorrer(b, pref="", out=None, prof=0):
    if out is None:
        out = {}
    i, fin = 0, len(b)
    while i < fin:
        try:
            k, i = _varint(b, i)
            fn, wt = k >> 3, k & 7
            p = f"{pref}.{fn}"
            if wt == 0:
                v, i = _varint(b, i); out.setdefault(p, []).append(v)
            elif wt == 1:
                out.setdefault(p, []).append(struct.unpack_from("<d", b, i)[0]); i += 8
            elif wt == 5:
                out.setdefault(p, []).append(struct.unpack_from("<f", b, i)[0]); i += 4
            elif wt == 2:
                n, i = _varint(b, i); sub = b[i:i + n]
                if prof < 6 and n > 1 and _parece_mensaje(sub):
                    _recorrer(sub, p, out, prof + 1)
                i += n
            else:
                return out
        except Exception:
            return out
    return out


def _con_signo(u):
    return u - (1 << 64) if u >= (1 << 63) else u


def leer_telemetria(ruta_video, verbose=True):
    """Telemetria por frame del MP4. No necesita ffmpeg ni el .proto de DJI."""
    offs = muestras_djmd(ruta_video)
    crudo = {}
    with open(ruta_video, "rb") as fh:
        for off, tam in offs:
            fh.seek(off)
            _recorrer(fh.read(tam), "", crudo)

    out = {}
    for ruta, (nombre, conv, signo) in _CAMPOS.items():
        vals = crudo.get(ruta)
        if not vals:
            continue
        out[nombre] = [conv(_con_signo(v) if signo else v) for v in vals]
    out["n_muestras"] = len(offs)
    if verbose:
        print(f"  telemetria DJI: {len(offs)} muestras | "
              f"campos: {', '.join(k for k in out if k != 'n_muestras')}")
    return out


def _mediana(a):
    s = sorted(a)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def frame_data_desde_video(ruta_video, focal_px=None, verbose=True):
    """`Config.frame_data` desde la telemetria, mas la deriva del dron durante el clip.
    """
    tel = leer_telemetria(ruta_video, verbose=False)
    fd = {}
    for k in ("lat_deg", "lon_deg", "absolute_altitude_m",
              "drone_yaw_deg", "gimbal_pitch_deg"):
        if k not in tel:
            raise ValueError(f"La telemetria no trae '{k}'. Revisa el modelo de dron.")
        fd[k] = _mediana(tel[k])
    fd["width"] = int(tel["width"][0]) if "width" in tel else None
    fd["height"] = int(tel["height"][0]) if "height" in tel else None
    if focal_px:
        fd["focal"] = float(focal_px)

    lat = fd["lat_deg"]
    dE = (max(tel["lon_deg"]) - min(tel["lon_deg"])) * 111320 * math.cos(math.radians(lat))
    dN = (max(tel["lat_deg"]) - min(tel["lat_deg"])) * 110540
    dZ = max(tel["absolute_altitude_m"]) - min(tel["absolute_altitude_m"])
    dYaw = max(tel["drone_yaw_deg"]) - min(tel["drone_yaw_deg"])
    dGim = max(tel["gimbal_pitch_deg"]) - min(tel["gimbal_pitch_deg"])
    deriva = dict(este_m=dE, norte_m=dN, vertical_m=dZ,
                  yaw_deg=dYaw, gimbal_deg=dGim,
                  horizontal_m=math.hypot(dE, dN), fps=tel.get("fps", [None])[0])

    if verbose:
        print(f"  pose (mediana de {tel['n_muestras']} muestras)")
        print(f"    lat {fd['lat_deg']:.7f}  lon {fd['lon_deg']:.7f}  "
              f"alt {fd['absolute_altitude_m']:.3f} m")
        print(f"    yaw dron {fd['drone_yaw_deg']:.1f}  "
              f"pitch gimbal {fd['gimbal_pitch_deg']:.1f}")
        print(f"  deriva en todo el clip: {deriva['horizontal_m']:.2f} m horizontal, "
              f"{dZ:.2f} m vertical, yaw {dYaw:.1f} deg, gimbal {dGim:.1f} deg")
    return fd, deriva, tel


_RE_TS_DJI = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")


def creation_time_mp4(ruta_video):
    """`creation_time` del box mvhd, en UTC. Devuelve None si no se puede leer."""
    import datetime
    try:
        with open(ruta_video, "rb") as fh:
            r = _buscar(fh, 0, os.path.getsize(ruta_video), ["moov", "mvhd"])
            if not r:
                return None
            fh.seek(r[0])
            b = fh.read(min(32, r[1] - r[0]))
            ver = b[0]
            t = (struct.unpack_from(">Q", b, 4)[0] if ver == 1
                 else struct.unpack_from(">I", b, 4)[0])
        if not t:
            return None
        return datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=t)
    except Exception:
        return None


def nombre_calibracion(video_nombre: str) -> str:
    """Nombre canonico de la calibracion del airblast para un video."""
    base = os.path.splitext(os.path.basename(video_nombre))[0]
    m = _RE_TS_DJI.search(base)
    sello = "".join(m.groups()) if m else base
    return f"calibracion_airblast_{sello}.pkl"


def sello_video(video_nombre: str) -> str | None:
    """El sello de tiempo DJI del nombre, o None."""
    m = _RE_TS_DJI.search(os.path.basename(video_nombre))
    return "".join(m.groups()) if m else None


def video_declarado(ruta: str) -> str | None:
    """El video que una calibracion declara POR DENTRO, o None si no lo dice."""
    try:
        c = normalizar_calibracion(leer_calibracion(ruta))
    except Exception:
        return None
    v = c.get("video")
    return os.path.basename(str(v)) if v else None


def calibracion_calza(ruta: str, video_nombre: str) -> bool | None:
    """Si esta calibracion es de este video: True, False, o None si no consta."""
    objetivo = os.path.splitext(os.path.basename(video_nombre))[0].lower()
    decl = video_declarado(ruta)
    if decl:
        return os.path.splitext(decl)[0].lower() == objetivo
    m = re.search(r"(?<!\d)(\d{14})(?!\d)", os.path.basename(ruta))
    if m:
        return m.group(1) == (sello_video(video_nombre) or "")
    return None


def migrar_nombre_calibracion(cfg, verbose=True) -> str | None:
    """Renombra al nombre canonico la calibracion de ESTE video que no lo use."""
    canonico = nombre_calibracion(cfg.video_nombre)
    destino = os.path.join(cfg.dir_fase(0), canonico)
    if os.path.exists(destino):
        return None
    for f in sorted(os.listdir(cfg.dir_fase(0))):
        if f == canonico or not re.search(r"calib", f, re.I):
            continue
        origen = os.path.join(cfg.dir_fase(0), f)
        if not os.path.isfile(origen) or video_declarado(origen) is None:
            continue
        if calibracion_calza(origen, cfg.video_nombre):
            os.rename(origen, destino)
            if verbose:
                print(f"    [calibracion] {f} declara este video por dentro "
                      f"-> renombrada a {canonico}")
            return canonico
    return None


def ts_inicio_desde_video(ruta_video, tz_local="America/Santiago", verbose=True):
    """Hora local del primer frame, del nombre del archivo."""
    import datetime
    from zoneinfo import ZoneInfo

    m = _RE_TS_DJI.search(os.path.basename(ruta_video))
    if not m:
        raise ValueError(
            f"El nombre '{os.path.basename(ruta_video)}' no tiene el patron "
            f"DJI_YYYYMMDDhhmmss. Pasa ts_inicio_video a mano.")
    loc = datetime.datetime(*(int(g) for g in m.groups()))
    ts = loc.isoformat(timespec="seconds")

    utc = creation_time_mp4(ruta_video)
    if utc is not None:
        esperado = loc.replace(tzinfo=ZoneInfo(tz_local)).astimezone(
            datetime.timezone.utc).replace(tzinfo=None)
        dif = abs((utc - esperado).total_seconds())
        if verbose:
            estado = "calza" if dif <= 5 else f"DISCREPA {dif:.0f} s"
            print(f"    ts del nombre {ts} ({tz_local}) vs MP4 "
                  f"{utc.isoformat()}Z -> {estado}")
        if dif > 60:
            print(f"    [aviso] el nombre y el contenedor difieren {dif/60:.0f} min. "
                  f"Si el archivo fue renombrado, manda el contenedor.")
    return ts


def leer_calibracion(ruta):
    """Calibracion del airblast. Acepta pickle o JSON pese a la extension."""
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        with open(ruta, "rb") as fh:
            return pickle.load(fh)


def normalizar_calibracion(c):
    """Unifica los dos esquemas de calibracion del airblast que circulan."""
    if not isinstance(c, dict):
        raise TypeError(f"Calibracion de tipo inesperado: {type(c).__name__}")
    if "PIVOT_X_NAT" in c or "PPM_NAT" in c:
        t0 = c.get("T0_SUBFRAME")
        return dict(esquema="v7.2", video=c.get("VIDEO") or c.get("video"),
                    blast_frame=int(round(float(t0))) if t0 is not None else None,
                    t0_subframe=float(t0) if t0 is not None else None,
                    pivot_x_nat=float(c["PIVOT_X_NAT"]),
                    pivot_y_nat=float(c["PIVOT_Y_NAT"]),
                    ppm_nat=float(c["PPM_NAT"]), crudo=c)
    if "pivots" in c and c["pivots"]:
        p = c["pivots"][0]
        return dict(esquema="gui_v6", video=c.get("video"),
                    blast_frame=(int(c["blast_frame"])
                                 if c.get("blast_frame") is not None else None),
                    t0_subframe=p.get("t0"),
                    pivot_x_nat=float(p["x"]), pivot_y_nat=float(p["y"]),
                    ppm_nat=float(p.get("ppm", c.get("pixels_per_meter"))),
                    crudo=c)
    raise KeyError("No reconozco el esquema de la calibracion. Claves: "
                   + str(list(c)))


def buscar_calibracion(dir_datos, video_nombre, verbose=True):
    """La calibracion de resultados/0_preprocesamiento/ que corresponde a ESTE video."""
    objetivo = os.path.splitext(os.path.basename(video_nombre))[0].lower()
    sello = sello_video(video_nombre)
    canonico = nombre_calibracion(video_nombre)

    def _leer(f):
        return normalizar_calibracion(leer_calibracion(os.path.join(dir_datos, f)))

    if os.path.exists(os.path.join(dir_datos, canonico)):
        try:
            c = _leer(canonico)
            if verbose:
                print(f"    calibracion: {canonico}   [nombre canonico]")
            return canonico, c, "nombre canonico del video"
        except Exception as e:
            if verbose:
                print(f"    [aviso] {canonico} existe pero no se puede leer "
                      f"({type(e).__name__}); sigo buscando")

    candidatos, por_sello, sin_video, otro_sello, vistos = [], [], [], [], []
    for f in sorted(os.listdir(dir_datos)):
        if not re.search(r"calib", f, re.I):
            continue
        try:
            c = _leer(f)
        except Exception as e:
            vistos.append((f, f"ilegible: {type(e).__name__}"))
            continue
        decl = str(c.get("video") or "")
        m_otro = re.search(r"(?<!\d)(\d{14})(?!\d)", f)
        vistos.append((f, decl or (f"sello {m_otro.group(1)} en el nombre"
                                   if m_otro else
                                   f"(esquema {c['esquema']}, sin campo VIDEO)")))
        if decl and os.path.splitext(decl)[0].lower() == objetivo:
            candidatos.append((f, c))
        elif decl:
            otro_sello.append((f, f"declara {decl}"))
        elif sello and m_otro and m_otro.group(1) == sello:
            por_sello.append((f, c))
        elif m_otro:
            otro_sello.append((f, f"sello {m_otro.group(1)}"))
        else:
            sin_video.append((f, c))

    if len(candidatos) > 1:
        raise ValueError(
            f"Hay {len(candidatos)} calibraciones que declaran el mismo video: "
            f"{[c[0] for c in candidatos]}. Deja una sola o eligela a mano.")
    if len(candidatos) == 1:
        if verbose:
            print(f"    calibracion: {candidatos[0][0]}   [declara el video "
                  f"por dentro]")
        return candidatos[0][0], candidatos[0][1], "campo VIDEO de la calibracion"

    if len(por_sello) == 1:
        if verbose:
            print(f"    calibracion: {por_sello[0][0]}   [sello {sello} en el "
                  f"nombre]")
        return por_sello[0][0], por_sello[0][1], f"sello {sello} en el nombre"
    if len(por_sello) > 1:
        raise ValueError(
            f"Hay {len(por_sello)} calibraciones con el sello {sello} en el "
            f"nombre: {[c[0] for c in por_sello]}. Deja una sola.")

    if len(sin_video) == 1 and not otro_sello:
        if verbose:
            print(f"    calibracion: {sin_video[0][0]}   [SUPUESTO: no declara "
                  f"video, no trae el sello, y es la unica en datos/]")
            print(f"      para que deje de ser un supuesto, renombrala a "
                  f"{canonico}")
        return sin_video[0][0], sin_video[0][1], "SUPUESTO: la unica en datos/"

    if verbose:
        det = "\n".join(f"      {f:46s} -> {d}" for f, d in vistos) or "      (ninguna)"
        print(f"    calibracion: NO la encuentro para "
              f"'{os.path.basename(video_nombre)}'\n{det}")
        if otro_sello:
            print(f"      hay {len(otro_sello)} que son de OTRO video "
                  f"({', '.join(d for _, d in otro_sello)}): NO se usan.")
        print(f"      la produce la celda C1b de la Fase 0, y la guardara como "
              f"{canonico}")
    return None, None, None


def resolver_evento(cfg, verbose=True):
    """Rellena todo lo derivable de `cfg.video_nombre`. Devuelve `procedencia`."""
    proc = {"video": os.path.basename(cfg.video_nombre)}
    try:
        st = os.stat(cfg.video)
        proc["video_bytes"] = st.st_size
    except OSError:
        pass

    if verbose:
        print(f"  resolviendo el evento desde {proc['video']}")

    if cfg.ts_inicio_video:
        proc["ts_inicio_video"] = "fijado a mano"
    else:
        cfg.ts_inicio_video = ts_inicio_desde_video(cfg.video, cfg.tz_local, verbose)
        proc["ts_inicio_video"] = "nombre del archivo"
    if verbose:
        print(f"    ts_inicio_video = {cfg.ts_inicio_video}")

    if cfg.frame_data:
        proc["frame_data"] = "fijado a mano"
        deriva = None
    else:
        fd, deriva, _ = frame_data_desde_video(cfg.video, verbose=False)
        ancho = fd.get("width") or 3840
        focal = cfg.focal_px_por_ancho.get(ancho)
        if focal is None:
            raise ValueError(
                f"No tengo la focal para un sensor de {ancho} px. Agrega "
                f"{ancho}: <focal_px> a Config.focal_px_por_ancho.")
        fd["focal"] = float(focal)
        cfg.frame_data = fd
        proc["frame_data"] = "telemetria djmd del MP4"
        proc["deriva_dron"] = {k: round(v, 3) for k, v in deriva.items()
                               if isinstance(v, (int, float))}
    if verbose:
        fd = cfg.frame_data
        print(f"    camara: lat {fd['lat_deg']:.6f}  lon {fd['lon_deg']:.6f}  "
              f"alt {fd['absolute_altitude_m']:.1f} m")
        print(f"            yaw {fd['drone_yaw_deg']:.1f}  "
              f"gimbal {fd['gimbal_pitch_deg']:.1f}  focal {fd['focal']:.0f} px")
        if deriva:
            print(f"    deriva del dron en todo el clip: "
                  f"{deriva['horizontal_m']:.2f} m horizontal, "
                  f"{deriva['vertical_m']:.2f} m vertical, "
                  f"yaw {deriva['yaw_deg']:.1f} deg")
            if deriva["horizontal_m"] > 2.0 or deriva["yaw_deg"] > 3.0:
                print("    [AVISO] el dron NO esta fijo. La homografia de la "
                      "Fase 3 supone una sola pose: ese movimiento entra "
                      "directo en la velocidad.")

    migrar_nombre_calibracion(cfg, verbose=verbose)
    calib = None
    fijada = (cfg.calibracion_nombre and cfg.calibracion_nombre != "auto"
              and os.path.exists(cfg.calibracion))
    if fijada and calibracion_calza(cfg.calibracion, cfg.video_nombre) is False:
        if verbose:
            print(f"    [aviso] calibracion_nombre={cfg.calibracion_nombre} es "
                  f"de OTRO video: se ignora y se busca la que corresponde.")
        cfg.calibracion_nombre = "auto"
        fijada = False
    if fijada:
        calib = normalizar_calibracion(leer_calibracion(cfg.calibracion))
        proc["calibracion"] = f"fijada a mano: {cfg.calibracion_nombre}"
    else:
        nombre, calib, regla = buscar_calibracion(cfg.dir_fase(0),
                                                  cfg.video_nombre, verbose)
        if nombre:
            cfg.calibracion_nombre = nombre
            proc["calibracion"] = f"{nombre}  [{regla}]"
        else:
            proc["calibracion"] = (
                f"PENDIENTE: la produce C1b y la guardara como "
                f"{nombre_calibracion(cfg.video_nombre)}")
    if calib is not None:
        proc["calibracion_esquema"] = calib["esquema"]

    if cfg.pivot_frame is not None:
        proc["pivot_frame"] = "fijado a mano"
        if calib is not None and calib.get("blast_frame") is not None \
                and abs(int(calib["blast_frame"]) - int(cfg.pivot_frame)) > 1:
            print(f"    [AVISO] pivot_frame={cfg.pivot_frame} fijado a mano, "
                  f"pero la calibracion de este video pone la detonacion en "
                  f"f{int(calib['blast_frame'])}. Si el numero venia de otro "
                  f"video, ponlo en None y deja que se resuelva solo.")
    elif calib is not None and calib.get("blast_frame") is not None:
        cfg.pivot_frame = int(calib["blast_frame"])
        proc["pivot_frame"] = (
            f"{'blast_frame' if calib['esquema'] == 'gui_v6' else 'T0_SUBFRAME'}"
            f" de {cfg.calibracion_nombre}")
    else:
        proc["pivot_frame"] = "PENDIENTE: lo pone C0b (onset) o C1b (calibracion)"
    if verbose:
        print(f"    pivot_frame = {cfg.pivot_frame}   "
              f"({proc['pivot_frame']})")

    faltan = [n for n, r in (("DEM", cfg.dem), ("ortofoto", cfg.satelital))
              if not os.path.exists(r)]
    if faltan and verbose:
        print(f"    [aviso] faltan en datos/: {', '.join(faltan)}. "
              f"La Fase 3 puede correr igual en coordenadas de video, pero no "
              f"georreferencia.")
    proc["insumos_externos"] = {"dem": os.path.basename(cfg.dem_nombre),
                                "ortofoto": os.path.basename(cfg.satelital_nombre),
                                "bounds_mina": list(cfg.bounds_mina)}
    return proc


def refrescar_evento(cfg, verbose=True):
    """Vuelve a resolver lo que quedo pendiente y regraba config.json."""
    previo = (_leer_procedencia(cfg).get("fases", {}).get("0", {})
              .get("procedencia", {}))
    migrar_nombre_calibracion(cfg, verbose=verbose)
    if cfg.calibracion_nombre != "auto" and (
            not os.path.exists(cfg.calibracion)
            or calibracion_calza(cfg.calibracion, cfg.video_nombre) is False):
        cfg.calibracion_nombre = "auto"
    proc = resolver_evento(cfg, verbose=verbose)
    for k, v in previo.items():
        if isinstance(v, str) and str(proc.get(k, "")).startswith("fijado a mano") \
                and not v.startswith("fijado a mano"):
            proc[k] = v
        else:
            proc.setdefault(k, v)
    cfg.guardar_json()
    sellar(cfg, 0, dict(procedencia=proc))
    if verbose:
        print("  config.json actualizado")
    return proc


def ruta_procedencia(cfg):
    return os.path.join(cfg.dir_resultados, "procedencia.json")


def _leer_procedencia(cfg):
    r = ruta_procedencia(cfg)
    if not os.path.exists(r):
        return {}
    try:
        with open(r, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def sellar(cfg, fase, extra=None):
    """Anota que la fase `fase` corrio para el video actual."""
    d = _leer_procedencia(cfg)
    d["video"] = os.path.basename(cfg.video_nombre)
    d.setdefault("fases", {})[str(fase)] = dict(
        video=os.path.basename(cfg.video_nombre), **(extra or {}))
    os.makedirs(cfg.dir_resultados, exist_ok=True)
    with open(ruta_procedencia(cfg), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
    return d


def exigir_sello(cfg, fase, nombre=""):
    """Falla si la fase `fase` no corrio para el video actual."""
    d = _leer_procedencia(cfg)
    v = os.path.basename(cfg.video_nombre)
    s = d.get("fases", {}).get(str(fase))
    if s is None:
        raise RuntimeError(
            f"La Fase {fase} no ha corrido para este video ({v}).\n"
            f"    {nombre or 'El artefacto que necesitas'} no existe o es de "
            f"otra corrida. Corre el notebook de la Fase {fase}.")
    if s.get("video") != v:
        raise RuntimeError(
            f"La Fase {fase} corrio para '{s.get('video')}' y ahora el video "
            f"es '{v}'.\n"
            f"    Sus artefactos son de otro evento: vuelve a correr la "
            f"Fase {fase}.")
    return s


def invalidar_si_cambio_video(cfg, verbose=True):
    """Si el video cambio, borra el cache y los sellos de fase."""
    d = _leer_procedencia(cfg)
    antes = d.get("video")
    ahora = os.path.basename(cfg.video_nombre)
    if antes is None or antes == ahora:
        return False
    if verbose:
        print(f"  [video distinto] antes '{antes}' -> ahora '{ahora}'")
    cache = os.path.join(cfg.dir_resultados, "cache")
    if os.path.isdir(cache):
        n = sum(len(f) for _, _, f in os.walk(cache))
        shutil.rmtree(cache, ignore_errors=True)
        if verbose:
            print(f"    cache borrado ({n} archivos): se regenera solo")
    with open(ruta_procedencia(cfg), "w", encoding="utf-8") as fh:
        json.dump({"video": ahora, "fases": {}}, fh, indent=2, ensure_ascii=False)
    if verbose:
        print("    sellos de fase limpiados: hay que volver a correr 1..5")
    return True
