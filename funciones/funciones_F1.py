"""Fase 1: segmentacion de la columna de gas. CLIPSeg y la novedad contra el fondo
localizan la columna y siembran a SAM 2, que la propaga en el video. Incluye las
mascaras de sombra y las validaciones de la segmentacion.
"""

import gc
import os
import time
import warnings

import cv2
import numpy as np

import funciones_F0 as F0

warnings.filterwarnings("ignore", message=r"cannot import name '_C' from 'sam2'")


PROMPTS_NUBE = ["white dust cloud", "dark gray smoke plume"]
PROMPTS_FONDO = ["bare rocky terrain", "dirt road", "mountain slope",
                 "dark shadow on the ground"]


class DetectorCLIPSeg:
    """Mapas continuos `heat_nube` y `ventaja` a la resolucion de entrada."""

    def __init__(self, modelo_id="CIDAS/clipseg-rd64-refined", dir_local=None,
                 prompts_nube=None, prompts_fondo=None):
        """Carga desde `dir_local` si existe; si no, descarga `modelo_id` y lo guarda ahi."""
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self._torch = torch
        self.prompts_nube = list(prompts_nube or PROMPTS_NUBE)
        self.prompts_fondo = list(prompts_fondo or PROMPTS_FONDO)
        self._todos = self.prompts_nube + self.prompts_fondo
        self._n_nube = len(self.prompts_nube)
        local = dir_local is not None and os.path.isdir(dir_local)
        fuente = dir_local if local else modelo_id
        self.processor = CLIPSegProcessor.from_pretrained(fuente)
        self.model = CLIPSegForImageSegmentation.from_pretrained(fuente)
        if dir_local is not None and not local:
            self.processor.save_pretrained(dir_local)
            self.model.save_pretrained(dir_local)
            print(f"CLIPSeg guardado en {F0.ruta_corta(dir_local)}")
        self.model.eval()
        print(f"CLIPSeg listo | nube: {self.prompts_nube} | "
              f"fondo: {self.prompts_fondo}")

    def mapas(self, small):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        inp = self.processor(text=self._todos, images=[pil] * len(self._todos),
                             padding=True, return_tensors="pt")
        with self._torch.no_grad():
            h = self._torch.sigmoid(self.model(**inp).logits).numpy()
        H, W = small.shape[:2]
        heat = cv2.resize(h[:self._n_nube].max(axis=0), (W, H))
        vent = cv2.resize(h[:self._n_nube].max(axis=0)
                          - h[self._n_nube:].max(axis=0), (W, H))
        return heat, vent


def video_overlay(ruta_video, masks, ruta_salida, fps, escala=0.5, alfa=0.45,
                  color=(0, 0, 255), etiqueta="SAM2"):
    fkeys = sorted(masks)
    H, W = masks[fkeys[0]].shape
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps, W, H)
    with F0.LectorVideo(ruta_video, escala) as lec:
        for fi, small in lec.recorrer(fkeys):
            small = small[:Hv, :Wv]
            m = masks[fi][:Hv, :Wv]
            capa = small.copy()
            capa[m > 0] = color
            vis = cv2.addWeighted(capa, alfa, small, 1 - alfa, 0)
            cv2.putText(vis, f"{etiqueta} f{fi} cob:{m.mean()*100:.1f}%",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(vis)
    vw.release()
    print(f"[video] {F0.ruta_corta(ruta_salida)}")
    return ruta_salida


def fondo_crudo(ruta_video, pivot_frame, escala=0.5, ventana=None, rango=None):
    """Mediana temporal del gris CRUDO en la ventana tranquila."""
    if rango is not None:
        ini, fin = int(rango[0]), int(rango[1])
        etiqueta = "ventana tranquila de la calibracion"
    else:
        ventana = F0.BG_WINDOW_DEFECTO if ventana is None else ventana
        ini, fin = max(0, pivot_frame - ventana), pivot_frame
        etiqueta = "N frames antes del pivote"
    pila = []
    with F0.LectorVideo(ruta_video, escala) as lec:
        for _fi, small in lec.recorrer(range(ini, fin)):
            pila.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if len(pila) < 5:
        raise RuntimeError(f"Ventana tranquila insuficiente [f{ini}, f{fin}): "
                           f"{len(pila)} frames.")
    n_leidos = len(pila)
    V = np.asarray(pila, np.float32)
    del pila
    fondo = np.median(V, axis=0).astype(np.float32)
    ruido = float(np.median(np.median(np.abs(V - fondo), axis=0)))
    del V
    gc.collect()
    print(f"[fondo crudo] {int(fondo.shape[0])}x{int(fondo.shape[1])} de "
          f"[f{ini}, f{fin}) ({n_leidos} frames, {etiqueta}) "
          f"| ruido base = {ruido:.2f}")
    print(f"              -> novedad_min = {max(6.0 * ruido, 5.0):.1f} | "
          f"novedad_ancla = {max(25.0 * ruido, 25.0):.1f}")
    if ruido < 0.5:
        print("   AVISO: ruido base casi nulo. O el video esta muy comprimido, "
              "o la ventana cae sobre frames repetidos. Los umbrales de novedad "
              "quedaran fijados por sus pisos (novedad_piso / novedad_ancla_piso).")
    return fondo, ruido


def geometria_escena(cfg, w_work, h_work=None, verbose=True):
    """Geometria de la escena (pivote y GSD), con una sola fuente de verdad."""
    ruta = cfg.art_geometria
    geo = None
    if os.path.exists(ruta):
        geo = F0.cargar_geometria(ruta)
        origen = os.path.basename(ruta)
        if geo.get("esquema", 1) < F0.ESQUEMA_GEOMETRIA:
            print(f"   AVISO: {origen} es de un esquema anterior "
                  f"(v{geo.get('esquema', 1)} < v{F0.ESQUEMA_GEOMETRIA}): le "
                  f"faltan campos. Lo recalculo desde la calibracion; vuelve a "
                  f"correr la celda C2 de la Fase 0 para dejarlo al dia.")
            geo = None
        elif h_work is not None and \
                list(geo.get("forma_trabajo", [])) != [int(h_work), int(w_work)]:
            print(f"   AVISO: {origen} se escribio para "
                  f"{geo.get('forma_trabajo')} y ahora el cuadro de trabajo es "
                  f"[{h_work}, {w_work}]. Vuelve a correr la Fase 0.")
    if geo is None:
        alto = int(h_work) if h_work is not None else int(round(w_work * 9 / 16))
        geo = F0.resolver_geometria(cfg, w_work, alto, verbose=False)
        origen = ("calibracion (la Fase 0 no dejo geometria.json; corre su "
                  "celda C2)")
    if verbose:
        px, py = geo["pivote_px_trabajo"]
        print(f"[geo] fuente: {origen}")
        print(f"[geo] pivote {px:.0f}, {py:.0f} px de trabajo | "
              f"gsd {geo['gsd_trabajo']:.5f} m/px | "
              f"cuadro {w_work * geo['gsd_trabajo']:.0f} m de ancho")
    return geo, origen


def resolver_pivote_px(cfg, w_work, h_work, manual=None, geo=None):
    """(px, py) del pivote a resolucion de TRABAJO."""
    if manual is not None:
        px, py = float(manual[0]), float(manual[1])
        print(f"[pivote] manual = ({px:.0f}, {py:.0f}) px de trabajo")
        return px, py
    if geo is None:
        geo, _ = geometria_escena(cfg, w_work, h_work, verbose=False)
    px, py = (float(v) for v in geo["pivote_px_trabajo"])
    print(f"[pivote] ({px:.0f}, {py:.0f}) px de trabajo   "
          f"[{geo.get('origen', {}).get('pivote', 'geometria.json')}]")
    return px, py


def resolver_gsd(cfg, w_work, es_nativa=True, manual=None, geo=None):
    """GSD en m/px a resolucion de TRABAJO, con chequeo de plausibilidad."""
    if manual is not None:
        gsd = float(manual)
        origen = "manual"
    else:
        if geo is None:
            geo, _ = geometria_escena(cfg, w_work, verbose=False)
        gsd = float(geo["gsd_trabajo"])
        origen = geo.get("origen", {}).get("gsd", "geometria.json")
    print(f"[gsd] {gsd:.5f} m/px de trabajo  ({origen})")
    print(f"      ancho del cuadro = {w_work * gsd:.0f} m  "
          f"<- si esto no es plausible, revisa la calibracion")
    return gsd


PAR = dict(
    heat_siembra=0.5, n_pos=8, n_neg=8, pos_repartidos=False,
    multiescala=True, escalas_m=(None, 50.0, 30.0), lado_min_px=400,
    heat_norm_pct=99.0, heat_norm_ref=0.85, margen_crece=2.6, consenso_min=2,
    consenso_degradado=False,
    ventana_max_m=None,
    novedad_k=6.0, novedad_piso=5.0, novedad_k_ancla=25.0,
    novedad_ancla_piso=25.0, novedad_blur=5, novedad_norm=True,
    gain_limites=(0.80, 1.25),
    alinear_fondo=True, ecc_escala=0.25, ecc_iter=60, ecc_eps=1e-5, ecc_blur=5,
    r_pivote_m=60.0, neg_anillo_m=(60.0, 200.0),
)

PAR_SAM2 = dict(
    t_busqueda_ini=0.07, t_busqueda_fin=1.85, t_retroceso=2.50,
    clip_fuerte=4000, ancla_modo="solido", frac_solida=0.50,
    siembra_mask=True, resync_frac=0.50, umbral_vacio=0.5,
    chunk=50, solape=4, skip=2, area_min_semilla=200,
)


PRESETS = {
    "hibrido": (
        dict(ventana_max_m=80.0),
        dict(ancla_modo="primero", siembra_mask=False, resync_frac=0.0),
    ),
}


def preset(nombre):
    """(par_det, par_sam) del preset. Copias nuevas: se pueden mutar."""
    if nombre not in PRESETS:
        raise KeyError(f"Preset '{nombre}' desconocido. Hay: {sorted(PRESETS)}")
    det, sam = PRESETS[nombre]
    return dict(det), dict(sam)


class Detector:
    """CLIPSeg con consenso multiescala + novedad contra el fondo registrado."""

    def __init__(self, detector, forma, pivote_px, gsd, fondo, ruido_base,
                 par=None, verbose=True):
        self.det = detector
        self.H, self.W = forma
        self.px, self.py = float(pivote_px[0]), float(pivote_px[1])
        self.gsd = float(gsd)
        self.fondo = np.asarray(fondo, np.float32)
        self.ruido_base = float(ruido_base)
        p = dict(PAR)
        p.update(par or {})
        self.p = p

        if self.fondo.shape != (self.H, self.W):
            raise ValueError(f"fondo {self.fondo.shape} != forma {(self.H, self.W)}")

        self.novedad_min = max(p["novedad_k"] * self.ruido_base, p["novedad_piso"])
        self.novedad_ancla = max(p["novedad_k_ancla"] * self.ruido_base,
                                 p["novedad_ancla_piso"])
        self.r_piv_px = p["r_pivote_m"] / self.gsd
        self.neg_r0 = p["neg_anillo_m"][0] / self.gsd
        self.neg_r1 = p["neg_anillo_m"][1] / self.gsd

        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self.dist_piv = np.hypot(xx - self.px, yy - self.py).astype(np.float32)
        del yy, xx

        self._fondo_ecc = cv2.GaussianBlur(
            cv2.resize(self.fondo, None, fx=p["ecc_escala"], fy=p["ecc_escala"]),
            (0, 0), 2)
        self._warp_memo = {}
        self._aviso_consenso = False

        if verbose:
            print(f"novedad px > {self.novedad_min:.1f} | "
                  f"ancla (mediana) > {self.novedad_ancla:.1f}")
            print(f"pivote ({self.px:.0f}, {self.py:.0f}) | "
                  f"r={p['r_pivote_m']:.0f} m = {self.r_piv_px:.0f} px | "
                  f"anillo negativo {self.neg_r0:.0f}-{self.neg_r1:.0f} px")
            if self.r_piv_px > 0.5 * np.hypot(self.W, self.H):
                print(f"   AVISO: r_pivote_m cubre todo el cuadro "
                      f"({self.W*self.gsd:.0f} m de ancho): el filtro de pivote "
                      f"queda INERTE. Reducelo.")
            if p["multiescala"]:
                et = ["completo" if t is None else
                      f"+-{t:.0f}m ({int(2*t/self.gsd)}px)" for t in p["escalas_m"]]
                print(f"consenso multiescala: {' | '.join(et)} | "
                      f"piso {p['lado_min_px']} px")

    def _fondo_alineado(self, g, devolver_warp=False):
        """Fondo limpio registrado sobre el frame `g` (gris, trabajo)."""
        p = self.p
        if not p["alinear_fondo"]:
            I = np.eye(2, 3, dtype=np.float32)
            return (self.fondo, I) if devolver_warp else self.fondo
        key = (round(float(g.mean()), 4), round(float(g.std()), 4),
               round(float(g[::97, ::97].sum()), 2))
        Wm = self._warp_memo.get(key)
        if Wm is None:
            gs = cv2.GaussianBlur(
                cv2.resize(g, None, fx=p["ecc_escala"], fy=p["ecc_escala"]),
                (0, 0), 2)
            Wm = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    gs, self._fondo_ecc, Wm, cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     p["ecc_iter"], p["ecc_eps"]), None, p["ecc_blur"])
            except cv2.error:
                Wm = np.eye(2, 3, dtype=np.float32)
            Wm[0, 2] /= p["ecc_escala"]
            Wm[1, 2] /= p["ecc_escala"]
            if len(self._warp_memo) > 96:
                self._warp_memo.clear()
            self._warp_memo[key] = Wm
        fa = cv2.warpAffine(self.fondo, Wm, (self.W, self.H),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        return (fa, Wm) if devolver_warp else fa

    def _ajuste_fotometrico(self, g, fondo):
        q = np.percentile(g, [25.0, 75.0]).astype(np.float32)
        fq = np.percentile(fondo, [25.0, 75.0]).astype(np.float32)
        rango = float(q[1] - q[0])
        if rango < 1e-3:
            return 1.0, 0.0
        gain = float(np.clip((fq[1] - fq[0]) / rango, *self.p["gain_limites"]))
        return gain, float(fq[0] - gain * q[0])

    def novedad(self, small, devolver_ajuste=False):
        """|gris normalizado - fondo REGISTRADO|, suavizado contra moteado."""
        p = self.p
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fondo, Wm = self._fondo_alineado(g, devolver_warp=True)
        gain, off = self._ajuste_fotometrico(g, fondo) if p["novedad_norm"] \
            else (1.0, 0.0)
        nov = np.abs((gain * g + off) - fondo)
        if p["novedad_blur"] and p["novedad_blur"] >= 3:
            nov = cv2.medianBlur(np.clip(nov, 0, 255).astype(np.uint8),
                                 p["novedad_blur"]).astype(np.float32)
        if devolver_ajuste:
            dx, dy = float(Wm[0, 2]), float(Wm[1, 2])
            ang = float(np.degrees(np.arctan2(Wm[1, 0], Wm[0, 0])))
            return nov, gain, off, (dx, dy, ang)
        return nov

    def _ventana(self, tam_m, r_eq_px=None):
        """Recorte cuadrado centrado en el pivote, de semilado `tam_m` metros (None =
        cuadro completo).
        """
        p = self.p
        if tam_m is None:
            return 0, 0, self.W, self.H
        r = tam_m / self.gsd
        if r_eq_px is not None:
            r = max(r, p["margen_crece"] * r_eq_px)
        if p["ventana_max_m"]:
            r = min(r, p["ventana_max_m"] / self.gsd)
        r = max(r, p["lado_min_px"] / 2.0)
        x0, y0 = max(0, int(self.px - r)), max(0, int(self.py - r))
        x1, y1 = min(self.W, int(self.px + r)), min(self.H, int(self.py + r))
        if (x1 - x0) < p["lado_min_px"] or (y1 - y0) < p["lado_min_px"]:
            return 0, 0, self.W, self.H
        return x0, y0, x1, y1

    def heat_consenso(self, small, r_eq_px=None, devolver_capas=False):
        """Heat por consenso (minimo) entre escalas. Devuelve (heat, vent,
        n_escalas_por_pixel).
        """
        p = self.p
        if not p["multiescala"]:
            h, v = self.det.mapas(small)
            u = np.ones_like(h, np.uint8)
            return (h, v, u, [(h, "completo")]) if devolver_capas else (h, v, u)
        capas, cobs, etiquetas, vistos = [], [], [], set()
        vent_full = None
        for tam in p["escalas_m"]:
            x0, y0, x1, y1 = self._ventana(tam, r_eq_px)
            if (x0, y0, x1, y1) in vistos:
                continue
            vistos.add((x0, y0, x1, y1))
            h_c, v_c = self.det.mapas(small[y0:y1, x0:x1])
            pc = float(np.percentile(h_c, p["heat_norm_pct"]))
            h_c = h_c / max(pc, 1e-3) * p["heat_norm_ref"]
            Hm = np.zeros((self.H, self.W), np.float32)
            Cm = np.zeros((self.H, self.W), bool)
            Hm[y0:y1, x0:x1] = h_c
            Cm[y0:y1, x0:x1] = True
            capas.append(Hm)
            cobs.append(Cm)
            etiquetas.append((Hm, "completo" if tam is None else f"+-{tam:.0f}m"))
            if tam is None:
                vent_full = np.zeros((self.H, self.W), np.float32)
                vent_full[y0:y1, x0:x1] = v_c
        _H, _C = np.stack(capas), np.stack(cobs)
        ncob = _C.sum(0).astype(np.uint8)
        heat = np.where(_C, _H, np.inf).min(0)
        heat[ncob == 0] = 0.0
        if len(capas) < p["consenso_min"] and not self._aviso_consenso:
            self._aviso_consenso = True
            print(f"   AVISO: las escalas {p['escalas_m']} colapsaron a "
                  f"{len(capas)} recorte(s) distinto(s), menos que "
                  f"consenso_min={p['consenso_min']}: el heat queda en cero "
                  f"mientras dure.")
        cmin = min(p["consenso_min"], len(capas)) if p["consenso_degradado"] \
            else p["consenso_min"]
        heat[ncob < cmin] = 0.0
        if vent_full is None:
            _, vent_full = self.det.mapas(small)
        return (heat, vent_full, ncob, etiquetas) if devolver_capas \
            else (heat, vent_full, ncob)

    def detectar_columna(self, small, umbral=None, usar_novedad=True,
                         r_eq_px=None):
        """Mayor componente conexa de (heat > umbral) & (novedad > umbral) cercana al
        pivote. Devuelve (mask, heat, vent).
        """
        umbral = self.p["heat_siembra"] if umbral is None else umbral
        heat, vent, _ = self.heat_consenso(small, r_eq_px)
        mask = (heat > umbral)
        if usar_novedad:
            mask = mask & (self.novedad(small) > self.novedad_min)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        if n <= 1:
            return np.zeros_like(mask), heat, vent
        iy = int(np.clip(self.py, 0, self.H - 1))
        ix = int(np.clip(self.px, 0, self.W - 1))
        cand = []
        for i in range(1, n):
            cx, cy = cent[i]
            contiene = bool(mask[iy, ix]) and lab[iy, ix] == i
            if contiene or np.hypot(cx - self.px, cy - self.py) <= self.r_piv_px:
                cand.append((stats[i, cv2.CC_STAT_AREA], i))
        if not cand:
            return np.zeros_like(mask), heat, vent
        return (lab == max(cand)[1]), heat, vent

    def sembrar(self, small, r_eq_px=None):
        """Puntos del prompt. Positivos: mayor novedad o, con `pos_repartidos`, farthest
        point sampling sobre el nucleo erosionado y mas claro que el fondo. Negativos:
        heat alto sin novedad en un anillo alrededor del pivote.
        """
        if small is None:
            return None
        p = self.p
        mask_col, heat, vent = self.detectar_columna(small, r_eq_px=r_eq_px)
        if mask_col.sum() < 20:
            return None
        nov = self.novedad(small)
        if p["pos_repartidos"]:
            nucleo = cv2.erode(mask_col.astype(np.uint8), np.ones((5, 5), np.uint8),
                               iterations=2) > 0
            if nucleo.sum() < p["n_pos"]:
                nucleo = mask_col.astype(bool)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            fondo = self._fondo_alineado(g)
            gain, off = self._ajuste_fotometrico(g, fondo)
            claro = nucleo & ((gain * g + off) > fondo)
            if claro.sum() >= p["n_pos"]:
                nucleo = claro
            pos = _fps_puntos(nucleo, p["n_pos"],
                              np.random.default_rng(0)).astype(int)
        else:
            ys, xs = np.where(mask_col)
            idx_top = np.argsort(nov[ys, xs])[-p["n_pos"]:]
            pos = np.stack([xs[idx_top], ys[idx_top]], axis=1)

        duros = ((heat > p["heat_siembra"]) & (nov <= self.novedad_min) &
                 (self.dist_piv > self.neg_r0) & (self.dist_piv < self.neg_r1) &
                 (~mask_col))
        bg = duros if duros.sum() > 50 else (vent < -0.5)
        neg = np.empty((0, 2), int)
        if bg.sum() > 50:
            yb, xb = np.where(bg)
            sel = np.random.choice(len(yb), size=min(p["n_neg"], len(yb)),
                                   replace=False)
            neg = np.stack([xb[sel], yb[sel]], axis=1)
        coords = np.vstack([pos, neg]).astype(np.float32)
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        return coords, labels

    @staticmethod
    def r_eq(mask):
        """Radio equivalente en px (para dimensionar el recorte del consenso)."""
        a = int(np.asarray(mask).sum())
        return float(np.sqrt(a / np.pi)) if a > 0 else None

    def chequeo_registro(self, lector, frames):
        """Fraccion del cuadro sobre el umbral de novedad en frames previos a la detonacion
        (debe ser < 10 %).
        """
        print("[chequeo] registro ECC: "
              f"{'activo' if self.p['alinear_fondo'] else 'DESACTIVADO'}")
        for fi in frames:
            small = lector[fi]
            if small is None:
                continue
            nov, g, o, (dx, dy, ang) = self.novedad(small, devolver_ajuste=True)
            print(f"[chequeo] f{fi}: shift=({dx:+.1f}, {dy:+.1f}) px {ang:+.2f}deg "
                  f"| gain={g:.3f} | "
                  f"{(nov > self.novedad_min).mean()*100:.1f}% sobre umbral")
        print("   Objetivo: <10% en todos.")


class Segmentador:
    """SAM 2 en modo video: reversa desde el anclaje hasta el nacimiento y propagacion
    hacia adelante por bloques.
    """

    def __init__(self, ckpt, cfg, dispositivo="cpu"):
        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(cfg, ckpt, device=dispositivo)
        print("SAM 2 listo.")

    def _sembrar_mascara(self, state, frame_semilla, masks_previas, forma,
                         area_min):
        m = masks_previas.get(frame_semilla)
        if m is None:
            return False, f"f{frame_semilla} sin mascara previa"
        area = int(m.sum())
        if area < area_min:
            return False, f"f{frame_semilla} mascara previa muy chica ({area} px)"
        if m.shape != forma:
            m = cv2.resize(m.astype(np.uint8), (forma[1], forma[0]),
                           interpolation=cv2.INTER_NEAREST)
        self.predictor.add_new_mask(inference_state=state, frame_idx=0,
                                    obj_id=1, mask=m.astype(bool))
        return True, (f"mascara de f{frame_semilla} ({area} px, "
                      f"{m.mean()*100:.1f}% cob)")

    def _sembrar_mask_col(self, state, small, fi, det, r_eq_px, area_min,
                          local=0):
        if small is None:
            return False, f"f{fi} no se pudo leer", 0
        mc, _, _ = det.detectar_columna(small, r_eq_px=r_eq_px)
        area = int(mc.sum())
        if area < area_min:
            return False, f"f{fi} mask_col muy chica ({area} px)", area
        self.predictor.add_new_mask(inference_state=state, frame_idx=local,
                                    obj_id=1, mask=mc.astype(bool))
        return True, f"mask_col de f{fi} ({area} px, {mc.mean()*100:.1f}% cob)", area

    def _sembrar_puntos_bloque(self, state, mapa_local, lector, det, r_eq_px):
        for local in range(len(mapa_local)):
            pts = det.sembrar(lector[mapa_local[local]], r_eq_px)
            if pts is not None:
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=local, obj_id=1,
                    points=pts[0], labels=pts[1])
                return True, (f"puntos CLIPSeg en f{mapa_local[local]} "
                              f"({len(pts[0])} pts)")
        return False, "CLIPSeg no encontro columna en el bloque"

    def segmentar(self, ruta_video, det, forma, dir_frames, pivot_frame,
                  analisis_fin, fps, escala=0.5, par=None):
        """Devuelve ({frame: mascara uint8}, info)."""
        p = dict(PAR_SAM2)
        p.update(par or {})
        H, W = forma
        skip = max(1, int(p["skip"]))

        busq_ini = pivot_frame + round(p["t_busqueda_ini"] * fps)
        busq_fin = pivot_frame + round(p["t_busqueda_fin"] * fps)
        max_retro = round(p["t_retroceso"] * fps)
        print(f"[ventana] ancla en [f{busq_ini}, f{busq_fin}) | retroceso "
              f"{max_retro} frames ({p['t_retroceso']:.2f} s @ {fps:.2f} fps)")

        lector = F0.LectorVideo(ruta_video, escala)
        try:
            print(f"Buscando ancla... (clip_fuerte={p['clip_fuerte']} px, "
                  f"novedad_ancla={det.novedad_ancla:.1f}, modo={p['ancla_modo']})")
            candidatos, area_max, req = [], 0, None
            for fi in range(busq_ini, busq_fin):
                small = lector[fi]
                if small is None:
                    break
                mc, _, _ = det.detectar_columna(small, r_eq_px=req)
                area = int(mc.sum())
                if area > 0:
                    req = det.r_eq(mc)
                area_max = max(area_max, area)
                if area < p["clip_fuerte"]:
                    continue
                nov = float(np.median(det.novedad(small)[mc.astype(bool)]))
                candidatos.append((fi, area, nov, nov >= det.novedad_ancla))

            aceptados = [c for c in candidatos if c[3]]
            ancla = None
            if aceptados:
                if p["ancla_modo"] == "mejor":
                    ancla = max(aceptados, key=lambda c: c[1])[0]
                elif p["ancla_modo"] == "solido":
                    amax = max(c[1] for c in aceptados)
                    ancla = next(c[0] for c in aceptados
                                 if c[1] >= p["frac_solida"] * amax)
                else:
                    ancla = aceptados[0][0]

            if candidatos:
                print(f"\n{'frame':>7} {'area px':>9} {'novedad':>9}  veredicto")
                for fi_, a_, n_, ok_ in candidatos:
                    mk = " <- ANCLA" if fi_ == ancla else ""
                    print(f"{fi_:>7} {a_:>9} {n_:>9.1f}  "
                          f"{'acepta' if ok_ else 'RECHAZA (novedad baja)'}{mk}")
            if ancla is None:
                if not candidatos:
                    raise RuntimeError(
                        f"Sin candidatos en [f{busq_ini}, f{busq_fin}): ningun "
                        f"frame supero clip_fuerte={p['clip_fuerte']} px (area "
                        f"maxima: {area_max} px). Con el consenso multiescala "
                        f"las areas bajan; revisa con ver_siembra() antes de "
                        f"bajar el umbral.")
                raise RuntimeError(
                    f"Sin ancla valida: {len(candidatos)} superaron clip_fuerte "
                    f"pero ninguno novedad_ancla={det.novedad_ancla:.1f} "
                    f"(mejor mediana = {max(c[2] for c in candidatos):.1f}).")

            mc_anc, _, _ = det.detectar_columna(lector[ancla])
            req_anc = det.r_eq(mc_anc)
            print(f"  ancla = f{ancla} (mask_col {int(mc_anc.sum())} px"
                  + (f", r_eq {req_anc:.0f} px)" if req_anc else ")"))

            inicio = max(0, ancla - max_retro, pivot_frame)
            if ancla - max_retro < pivot_frame:
                print(f"  reversa recortada por el pivote: "
                      f"f{ancla - max_retro} -> f{pivot_frame}")
            frames_inv = list(range(inicio, ancla + 1))[::-1]
            lm = F0.extraer_frames_jpeg(ruta_video, frames_inv, dir_frames, escala)
            state = self.predictor.init_state(video_path=dir_frames)
            if p["siembra_mask"]:
                okA, motA, _ = self._sembrar_mask_col(
                    state, lector[ancla], ancla, det, req_anc,
                    p["area_min_semilla"])
            else:
                pts = det.sembrar(lector[ancla], req_anc)
                okA = pts is not None
                motA = f"{len(pts[0])} puntos" if okA else "sin puntos"
                if okA:
                    self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=1,
                        points=pts[0], labels=pts[1])
            if not okA:
                raise RuntimeError(f"No se pudo sembrar en la ancla f{ancla}: {motA}")
            print(f"Reversa: siembra = {motA}")
            masks_rev = {}
            for li, _, ml in self.predictor.propagate_in_video(state):
                masks_rev[lm[li]] = (ml[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
            self.predictor.reset_state(state)
            del state
            gc.collect()
            for fi in sorted(masks_rev):
                if masks_rev[fi].mean() * 100 < p["umbral_vacio"]:
                    del masks_rev[fi]
                else:
                    break
            nacimiento = min(masks_rev) if masks_rev else ancla
            print(f"Reversa: [f{nacimiento}, f{ancla}] ({len(masks_rev)} frames)")

            frames_fwd = list(range(ancla, analisis_fin, skip))
            masks_fwd, log = {}, []
            t0, pos_i, cid = time.time(), 0, 0
            while pos_i < len(frames_fwd):
                ini = pos_i if cid == 0 else max(0, pos_i - p["solape"])
                fin = min(len(frames_fwd), pos_i + p["chunk"])
                bloque = frames_fwd[ini:fin]
                if not bloque:
                    break
                lm = F0.extraer_frames_jpeg(ruta_video, bloque, dir_frames, escala)
                state = self.predictor.init_state(video_path=dir_frames)

                m_prev = masks_fwd.get(bloque[0])
                req_b = det.r_eq(m_prev) if m_prev is not None else req_anc

                if cid == 0:
                    if p["siembra_mask"]:
                        ok, motivo, _ = self._sembrar_mask_col(
                            state, lector[bloque[0]], bloque[0], det, req_b,
                            p["area_min_semilla"])
                        tipo = "mask_col (bloque 0)"
                        if not ok:
                            ok, motivo = self._sembrar_puntos_bloque(
                                state, lm, lector, det, req_b)
                            tipo = "puntos (respaldo)"
                    else:
                        ok, motivo = self._sembrar_puntos_bloque(
                            state, lm, lector, det, req_b)
                        tipo = "puntos (bloque 0)"
                else:
                    mc0, _, _ = det.detectar_columna(lector[bloque[0]],
                                                     r_eq_px=req_b)
                    a_clip = int(mc0.sum())
                    a_prev = int(m_prev.sum()) if m_prev is not None else 0
                    if a_clip >= p["area_min_semilla"] and \
                            a_prev < p["resync_frac"] * a_clip:
                        self.predictor.add_new_mask(
                            inference_state=state, frame_idx=0, obj_id=1,
                            mask=mc0.astype(bool))
                        ok, tipo = True, "mask_col (RESYNC)"
                        motivo = (f"propagada {a_prev} px < "
                                  f"{p['resync_frac']:.0%} de mask_col {a_clip} px "
                                  f"en f{bloque[0]}")
                    else:
                        ok, motivo = self._sembrar_mascara(
                            state, bloque[0], masks_fwd, forma,
                            p["area_min_semilla"])
                        tipo = "mascara"
                        if not ok:
                            print(f"  aviso bloque {cid}: {motivo} -> respaldo CLIPSeg")
                            ok, motivo, _ = self._sembrar_mask_col(
                                state, lector[bloque[0]], bloque[0], det, req_b,
                                p["area_min_semilla"])
                            tipo = "mask_col (respaldo)"

                if not ok:
                    print(f"  bloque {cid} omitido ({motivo})")
                    log.append((cid, bloque[0], "ninguna", motivo))
                    self.predictor.reset_state(state)
                    del state
                    gc.collect()
                    pos_i, cid = fin, cid + 1
                    continue

                log.append((cid, bloque[0], tipo, motivo))
                for li, _, ml in self.predictor.propagate_in_video(state):
                    m = (ml[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
                    fi = lm[li]
                    if fi in masks_fwd and li < p["solape"] and cid > 0:
                        continue
                    masks_fwd[fi] = m
                self.predictor.reset_state(state)
                del state
                gc.collect()

                f0_, f1_ = lm[0], lm[len(lm) - 1]
                c0 = masks_fwd[f0_].mean() * 100 if f0_ in masks_fwd else np.nan
                c1 = masks_fwd[f1_].mean() * 100 if f1_ in masks_fwd else np.nan
                print(f"bloque {cid}: f{f0_}-f{f1_} | siembra={tipo} ({motivo}) "
                      f"| cob {c0:.1f}% -> {c1:.1f}% | "
                      f"{(time.time()-t0)/60:.1f} min")
                pos_i, cid = fin, cid + 1
        finally:
            lector.close()

        masks = {fi: m for fi, m in masks_rev.items() if fi < ancla}
        masks.update(masks_fwd)
        print(f"Fusion: {len(masks)} frames [f{min(masks)}, f{max(masks)}]")
        return masks, dict(ancla=ancla, nacimiento=nacimiento, log_siembra=log,
                           busqueda=(busq_ini, busq_fin), max_retroceso=max_retro)


_C_CLIP_RGB  = (255, 127, 14)
_C_SAM_RGB   = (31, 119, 180)
_C_AMBAS_RGB = (200, 200, 200)
_ALFA_RELLENO = 0.30


def _pintar_rgb(vis, m, color, alfa=_ALFA_RELLENO):
    """Relleno translucido sobre una imagen RGB, in place."""
    if m is None or not m.any():
        return vis
    vis[m] = ((1 - alfa) * vis[m] + alfa * np.array(color, float)).astype(np.uint8)
    return vis


def comparar_mascaras(frames, det, lector, masks, tol=2, fps=None,
                      pivot_frame=None):
    """CLIPSeg | SAM 2 | diferencia entre ambas, sobre el frame real."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(frames)
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.6 * n))
    ax = np.atleast_2d(ax)
    for r, fi in enumerate(frames):
        sm = lector[fi]
        if sm is None:
            continue
        mc, _, _ = det.detectar_columna(sm)
        ms = masks.get(fi)
        if ms is None:
            cand = [k for k in masks if abs(k - fi) <= tol]
            ms = masks[min(cand, key=lambda k: abs(k - fi))] if cand else None
        rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
        etq = f"f{fi}"
        if fps and pivot_frame is not None:
            etq += f"   (t = {(fi - pivot_frame)/fps:+.2f} s)"

        ax[r, 0].imshow(_pintar_rgb(rgb.copy(), mc, _C_CLIP_RGB))
        ax[r, 0].set_title(f"{etq}\nCLIPSeg — consenso + novedad: "
                           f"{int(mc.sum()):,} px", fontsize=9)

        mb = ms.astype(bool) if ms is not None else None
        o2 = rgb.copy()
        if mb is not None:
            _pintar_rgb(o2, mb, _C_SAM_RGB)
            ax[r, 1].set_title(f"SAM 2 — propagación temporal: "
                               f"{int(mb.sum()):,} px", fontsize=9)
        else:
            ax[r, 1].set_title("SAM 2 — sin máscara en este frame", fontsize=9)
        ax[r, 1].imshow(o2)

        o3 = rgb.copy()
        if mb is not None:
            solo_c, solo_s = mc & ~mb, mb & ~mc
            _pintar_rgb(o3, mc & mb, _C_AMBAS_RGB)
            _pintar_rgb(o3, solo_c, _C_CLIP_RGB)
            _pintar_rgb(o3, solo_s, _C_SAM_RGB)
            ax[r, 2].set_title(f"diferencia — solo CLIPSeg {int(solo_c.sum()):,} px"
                               f"   |   solo SAM 2 {int(solo_s.sum()):,} px",
                               fontsize=9)
        else:
            ax[r, 2].set_title("diferencia — no disponible", fontsize=9)
        ax[r, 2].imshow(o3)

        for c in range(3):
            ax[r, c].axis("off")

    fig.legend(handles=[
        Patch(facecolor=np.array(_C_CLIP_RGB) / 255, label="CLIPSeg"),
        Patch(facecolor=np.array(_C_SAM_RGB) / 255, label="SAM 2"),
        Patch(facecolor=np.array(_C_AMBAS_RGB) / 255, label="ambas")],
        loc="upper center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    plt.show()


PAR_SOMBRA_V6 = dict(
    lum_centro=80.0,  lum_pend=-0.08,
    sat_centro=30.0,  sat_pend=-0.10,
    cr_centro=120.0,  cr_pend=-0.06,
    grad_centro=20.0, grad_pend=-0.12,
    w_lum=0.35, w_sat=0.25, w_cr=0.20, w_grad=0.20,
)
UMBRAL_SOMBRA_V6 = 0.55
AREA_MIN_SOMBRA_V6 = 800
PRE_SOMBRA_V6 = 40


def _sigmoide(x, centro, pend):
    """[0,1] centrada en `centro`. pend<0: valores MENORES que centro -> 1."""
    z = -pend * (np.asarray(x, np.float32) - centro)
    return 1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0)))


def score_sombra(frame_bgr, fondo_bgr, lum_centro=80.0, lum_pend=-0.08,
                 sat_centro=30.0, sat_pend=-0.10, cr_centro=120.0,
                 cr_pend=-0.06, grad_centro=20.0, grad_pend=-0.12,
                 w_lum=0.35, w_sat=0.25, w_cr=0.20, w_grad=0.20):
    """Mapa de probabilidad de sombra en [0,1]. Alto = sombra."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    s_lum = _sigmoide(lab[:, :, 0], lum_centro, lum_pend)

    hsv_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_b = cv2.cvtColor(fondo_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_sat = _sigmoide(hsv_b[:, :, 1] - hsv_f[:, :, 1], sat_centro, sat_pend)

    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    s_cr = _sigmoide(ycc[:, :, 1], cr_centro, cr_pend)

    g_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_b = cv2.cvtColor(fondo_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e_f = (cv2.Sobel(g_f, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
           cv2.Sobel(g_f, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    e_b = (cv2.Sobel(g_b, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
           cv2.Sobel(g_b, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    perdida = np.sqrt(np.maximum(e_b - e_f, 0))
    s_grad = _sigmoide(perdida, grad_centro, grad_pend)

    score = (w_lum * s_lum + w_sat * s_sat + w_cr * s_cr + w_grad * s_grad)
    return score.astype(np.float32), dict(lum=s_lum, sat=s_sat, cr=s_cr,
                                          grad=s_grad, perdida=perdida)


def mascara_sombra(frame_bgr, fondo_bgr, umbral=UMBRAL_SOMBRA_V6,
                   area_min=AREA_MIN_SOMBRA_V6, par=None):
    """score_sombra + umbral + apertura/cierre 7x7 + filtro por area."""
    score, sen = score_sombra(frame_bgr, fondo_bgr, **(par or PAR_SOMBRA_V6))
    m = (score >= umbral).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, etq, est, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if est[i, cv2.CC_STAT_AREA] >= area_min:
            out[etq == i] = 255
    return out, score, sen


def fondo_crudo_color(ruta_video, pivot_frame, escala=0.5, ventana=None,
                      rango=None, verbose=True):
    """Mediana temporal del frame BGR CRUDO en la ventana tranquila."""
    if rango is not None:
        ini, fin = int(rango[0]), int(rango[1])
        etiqueta = "ventana tranquila de la calibracion"
    else:
        ventana = F0.BG_WINDOW_DEFECTO if ventana is None else ventana
        ini, fin = max(0, pivot_frame - ventana), pivot_frame
        etiqueta = "N frames antes del pivote"
    pila = []
    with F0.LectorVideo(ruta_video, escala) as lec:
        for _fi, small in lec.recorrer(range(ini, fin)):
            pila.append(small)
    if len(pila) < 5:
        raise RuntimeError(f"Ventana tranquila insuficiente [f{ini}, f{fin}): "
                           f"{len(pila)} frames.")
    V = np.asarray(pila, np.uint8)
    del pila
    fondo = np.median(V, axis=0).astype(np.uint8)
    del V
    gc.collect()
    if verbose:
        print(f"[fondo crudo color] {fondo.shape[0]}x{fondo.shape[1]} de "
              f"[f{ini}, f{fin}) ({etiqueta})")
    return fondo


def detectar_sombras(ruta_video, fondo_bgr, pivot_frame, fin, escala=0.5,
                     pre=PRE_SOMBRA_V6, skip=1, umbral=UMBRAL_SOMBRA_V6,
                     area_min=AREA_MIN_SOMBRA_V6, par=None,
                     desenfocar_fondo=False, verbose=True):
    """Mascara de sombra por frame, de `pivot_frame - pre` hasta `fin`."""
    if fin is None:
        raise ValueError("`fin` no puede ser None: pasa CFG.FIN o N_FRAMES.")
    ini = max(0, int(pivot_frame) - int(pre))
    fondo = cv2.GaussianBlur(fondo_bgr, (5, 5), 0) if desenfocar_fondo else fondo_bgr
    masks, cob = {}, []
    if verbose:
        print(f"[sombra multicanal] f{ini} .. f{int(fin)}  (pre={pre}, "
              f"skip={skip}, umbral={umbral}, area_min={area_min})")
    with F0.LectorVideo(ruta_video, escala) as lec:
        for fi, small in lec.recorrer(range(ini, int(fin), max(1, int(skip)))):
            if small.shape[:2] != fondo.shape[:2]:
                raise ValueError(
                    f"frame {small.shape[:2]} != fondo {fondo.shape[:2]}: el "
                    f"fondo tiene que venir de la MISMA escala de trabajo.")
            m, _, _ = mascara_sombra(cv2.GaussianBlur(small, (5, 5), 0), fondo,
                                     umbral=umbral, area_min=area_min, par=par)
            masks[int(fi)] = m
            cob.append(float((m > 0).mean()))
    if not masks:
        raise RuntimeError(f"Sin frames en [f{ini}, f{int(fin)}).")
    cob = np.asarray(cob)
    n_pre = sum(1 for f in masks if f < pivot_frame)
    if verbose:
        print(f"  {len(masks)} mascaras | cobertura media {cob.mean()*100:.2f}% "
              f"(min {cob.min()*100:.2f}%, max {cob.max()*100:.2f}%)")
        print(f"  frames pre-tronadura: {n_pre}  "
              f"{'(ok, la sombra estatica se puede construir)' if n_pre >= 10 else '<< POCOS: sube `pre`'}")
        if cob.mean() < 0.005:
            print("  AVISO: cobertura casi nula. Revisa que el fondo sea el "
                  "CRUDO en color y que la escala coincida.")
    return masks, dict(metodo="multicanal V6 (Arjomandi 2022)", umbral=umbral,
                       area_min=area_min, pre=int(pre), skip=int(skip),
                       escala=escala, ini=int(ini), fin=int(fin),
                       par=dict(par or PAR_SOMBRA_V6),
                       desenfocar_fondo=bool(desenfocar_fondo),
                       cobertura_media=float(cob.mean()), n_pre=int(n_pre))


def _bin(m):
    """Cualquier mascara (0/1, 0/255, bool) -> bool."""
    return np.asarray(m) > 0


def iou_mascaras(a, b) -> float:
    a, b = _bin(a), _bin(b)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def dice_mascaras(a, b) -> float:
    a, b = _bin(a), _bin(b)
    s = int(a.sum() + b.sum())
    return float(2.0 * (a & b).sum() / s) if s else 1.0


def _centroide(m):
    ys, xs = np.where(_bin(m))
    if len(xs) == 0:
        return (np.nan, np.nan)
    return (float(xs.mean()), float(ys.mean()))


def _n_componentes(m, area_min=50):
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        _bin(m).astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, 0.0
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    areas = areas[areas >= area_min]
    if len(areas) == 0:
        return 0, 0.0
    return int(len(areas)), float(areas[0] / areas.sum())


def _fps_puntos(mask, k, rng):
    """k puntos bien repartidos dentro de `mask` (farthest point sampling)."""
    ys, xs = np.where(_bin(mask))
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    P = np.stack([xs, ys], 1).astype(np.float32)
    if len(P) <= k:
        return P
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return P[idx]


def calidad_sam2(masks, ruta_video, ckpt, cfg, dispositivo="cpu", escala=0.5,
                 fps=30.0, pivot_frame=0, frames=None, n_muestras=40,
                 delta_logit=1.0, n_pos=6, n_neg=4, area_min_comp=50,
                 semilla=0, verbose=True):
    """Metrica interna de SAM 2 sobre mascaras ya calculadas."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    claves = sorted(masks)
    if frames is None:
        if n_muestras >= len(claves):
            frames = claves
        else:
            idx = np.linspace(0, len(claves) - 1, n_muestras).round().astype(int)
            frames = [claves[i] for i in sorted(set(idx.tolist()))]
    frames = [f for f in frames if f in masks]
    if not frames:
        raise ValueError("Ningun frame pedido esta en `masks`.")

    if verbose:
        print(f"[calidad SAM2] {len(frames)} frames de {len(claves)} "
              f"(f{frames[0]} .. f{frames[-1]}) en {dispositivo}")

    modelo = build_sam2(cfg, ckpt, device=dispositivo)
    pred = SAM2ImagePredictor(modelo)
    rng = np.random.default_rng(semilla)

    columnas = ["frame", "t_s", "area_px", "cobertura", "iou_predicho",
                "estabilidad", "iou_vs_video", "n_comp", "frac_princ",
                "frac_borde"]
    filas, t0 = [], time.time()

    with F0.LectorVideo(ruta_video, escala) as lec, torch.inference_mode():
        for k, fi in enumerate(frames):
            small = lec[fi]
            if small is None:
                continue
            m = _bin(masks[fi])
            if m.shape != small.shape[:2]:
                m = _bin(F0.normalizar_mascara(m.astype(np.uint8) * 255,
                                               small.shape[:2]))
            area = int(m.sum())
            H, W = m.shape
            if area < 20:
                filas.append([fi, (fi - pivot_frame) / fps, area,
                              area / (H * W), np.nan, np.nan, np.nan, 0, 0.0,
                              0.0])
                continue

            pred.set_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            ys, xs = np.where(m)
            caja = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
            nucleo = cv2.erode(m.astype(np.uint8),
                               np.ones((5, 5), np.uint8), iterations=2)
            if nucleo.sum() < n_pos:
                nucleo = m.astype(np.uint8)
            pos = _fps_puntos(nucleo, n_pos, rng)
            k_ext = max(3, int(round(0.10 * np.sqrt(area))))
            anillo = (cv2.dilate(m.astype(np.uint8), np.ones((k_ext, k_ext),
                                                             np.uint8)) > 0)
            anillo &= ~(cv2.dilate(m.astype(np.uint8),
                                   np.ones((3, 3), np.uint8)) > 0)
            neg = _fps_puntos(anillo, n_neg, rng)

            pts = np.concatenate([pos, neg], 0) if len(neg) else pos
            lab = np.concatenate([np.ones(len(pos), np.int32),
                                  np.zeros(len(neg), np.int32)])

            m_sam, iou_pred, logits = pred.predict(
                point_coords=pts, point_labels=lab, box=caja,
                multimask_output=False)

            lo = np.asarray(logits[0], np.float32)
            a_dur = float((lo > delta_logit).sum())
            a_blan = float((lo > -delta_logit).sum())
            estab = a_dur / a_blan if a_blan > 0 else 0.0

            m2 = _bin(m_sam[0])
            n_c, f_pr = _n_componentes(m, area_min_comp)
            met = F0.metricas_mascara(m.astype(np.uint8), area_min=1) or {}

            filas.append([fi, round((fi - pivot_frame) / fps, 3), area,
                          round(area / (H * W), 5),
                          round(float(iou_pred[0]), 4), round(estab, 4),
                          round(iou_mascaras(m, m2), 4), n_c, round(f_pr, 4),
                          round(float(met.get("frac_borde", 0.0)), 4)])

            if verbose and (k + 1) % 10 == 0:
                print(f"   {k+1}/{len(frames)}  ({(time.time()-t0)/60:.1f} min)")

    del pred, modelo
    gc.collect()

    A = np.array([[f[4], f[5], f[6], f[7]] for f in filas], float)
    val = ~np.isnan(A[:, 0])
    resumen = dict(
        n=int(val.sum()),
        iou_predicho_mediana=float(np.median(A[val, 0])) if val.any() else np.nan,
        iou_predicho_p10=float(np.percentile(A[val, 0], 10)) if val.any() else np.nan,
        estabilidad_mediana=float(np.median(A[val, 1])) if val.any() else np.nan,
        estabilidad_p10=float(np.percentile(A[val, 1], 10)) if val.any() else np.nan,
        iou_vs_video_mediana=float(np.median(A[val, 2])) if val.any() else np.nan,
        frames_multicomp=int((A[val, 3] > 1).sum()),
        frac_multicomp=float((A[val, 3] > 1).mean()) if val.any() else np.nan)

    if verbose:
        print(f"\n  iou_predicho  mediana {resumen['iou_predicho_mediana']:.3f}"
              f"  | p10 {resumen['iou_predicho_p10']:.3f}")
        print(f"  estabilidad   mediana {resumen['estabilidad_mediana']:.3f}"
              f"  | p10 {resumen['estabilidad_p10']:.3f}")
        print(f"  iou_vs_video  mediana {resumen['iou_vs_video_mediana']:.3f}"
              f"   <- acuerdo imagen/video del MISMO modelo")
        print(f"  frames con >1 componente: {resumen['frames_multicomp']} "
              f"({resumen['frac_multicomp']*100:.0f}%)")
    return filas, columnas, resumen


def _fabricar_lector(base, fn, escala):
    """Subclase de LectorVideo que aplica `fn` despues del reescalado."""
    class _LectorVar(base):
        def __init__(self, ruta, escala_ignorada=1.0):
            super().__init__(ruta, escala)

        def _escalar(self, frame):
            return fn(super()._escalar(frame))
    return _LectorVar


def _fabricar_extractor(fn, escala):
    """Reemplazo de F0.extraer_frames_jpeg que aplica `fn` antes de escribir."""
    def extraer(ruta_video, frames, carpeta, escala_ignorada=0.5):
        import shutil
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)
        os.makedirs(carpeta)
        mapa, cap = {}, cv2.VideoCapture(ruta_video)
        for local, fi in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            small = frame if escala == 1.0 else \
                cv2.resize(frame, None, fx=escala, fy=escala)
            cv2.imwrite(os.path.join(carpeta, f"{local:04d}.jpg"), fn(small))
            mapa[local] = fi
        cap.release()
        return mapa
    return extraer


class video_transformado:
    """Context manager: toda la Fase 1 lee el video ya editado."""

    def __init__(self, fn, escala):
        self.fn, self.escala = fn, escala

    def __enter__(self):
        self._lector, self._extraer = F0.LectorVideo, F0.extraer_frames_jpeg
        F0.LectorVideo = _fabricar_lector(self._lector, self.fn, self.escala)
        F0.extraer_frames_jpeg = _fabricar_extractor(self.fn, self.escala)
        return self

    def __exit__(self, *exc):
        F0.LectorVideo = self._lector
        F0.extraer_frames_jpeg = self._extraer
        return False


def _gamma(img, g):
    lut = (np.linspace(0, 1, 256) ** g * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _saturacion(img, k):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[..., 1] = np.clip(h[..., 1] * k, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _balance(img, gb, gg, gr):
    f = img.astype(np.float32) * np.array([gb, gg, gr], np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _ruido(img, sigma, semilla=0):
    rng = np.random.default_rng(semilla)
    r = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + r, 0, 255).astype(np.uint8)


def _clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _gris3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


_GEOM = {
    "id": (lambda im: im,
           lambda m: m,
           lambda x, y, H, W: (x, y),
           lambda H, W: (H, W)),
    "rot90": (lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
              lambda m: cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE),
              lambda x, y, H, W: (H - 1 - y, x),
              lambda H, W: (W, H)),
    "rot180": (lambda im: cv2.rotate(im, cv2.ROTATE_180),
               lambda m: cv2.rotate(m, cv2.ROTATE_180),
               lambda x, y, H, W: (W - 1 - x, H - 1 - y),
               lambda H, W: (H, W)),
    "rot270": (lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
               lambda m: cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE),
               lambda x, y, H, W: (y, W - 1 - x),
               lambda H, W: (W, H)),
    "espejo_h": (lambda im: cv2.flip(im, 1),
                 lambda m: cv2.flip(m, 1),
                 lambda x, y, H, W: (W - 1 - x, y),
                 lambda H, W: (H, W)),
    "espejo_v": (lambda im: cv2.flip(im, 0),
                 lambda m: cv2.flip(m, 0),
                 lambda x, y, H, W: (x, H - 1 - y),
                 lambda H, W: (H, W)),
}


def variante(nombre, geom="id", foto=None, escala_rel=1.0, escalar_umbrales=True,
             nota=""):
    """Una edicion del video. `escala_rel` es relativa a `CFG.escala`."""
    g = _GEOM[geom]
    foto = foto or (lambda im: im)
    return dict(nombre=nombre, geom=geom, escala_rel=float(escala_rel),
                escalar_umbrales=bool(escalar_umbrales), nota=nota,
                fn=lambda im: g[0](foto(im)),
                inv_mascara=g[1], punto=g[2], forma=g[3])


def catalogo_variantes(cuales=None) -> list:
    """Las ediciones disponibles. `cuales=None` -> el conjunto por defecto."""
    V = [
        variante("identidad", nota="control: tiene que dar IoU = 1.000"),

        variante("full", escala_rel=2.0, escalar_umbrales=True,
                 nota="4K nativo, umbrales en px reescalados por k^2 / k"),
        variante("full_umbral_fijo", escala_rel=2.0, escalar_umbrales=False,
                 nota="4K con los umbrales tal cual: mide el peso del desajuste"),
        variante("media_baja", escala_rel=0.7, escalar_umbrales=True,
                 nota="tercer punto del eje de resolucion (0.35 del nativo)"),

        variante("rot90", geom="rot90", nota="giro 90 horario"),
        variante("rot180", geom="rot180", nota="giro 180"),
        variante("rot270", geom="rot270", nota="giro 90 antihorario"),
        variante("espejo_h", geom="espejo_h", nota="espejo horizontal"),
        variante("espejo_v", geom="espejo_v", nota="espejo vertical"),

        variante("brillo_mas", foto=lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=30),
                 nota="+30 niveles"),
        variante("brillo_menos", foto=lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=-30),
                 nota="-30 niveles"),
        variante("contraste_alto", foto=lambda im: cv2.convertScaleAbs(im, alpha=1.30, beta=-38),
                 nota="ganancia 1.30 con media preservada"),
        variante("contraste_bajo", foto=lambda im: cv2.convertScaleAbs(im, alpha=0.75, beta=32),
                 nota="ganancia 0.75 con media preservada"),
        variante("gamma_baja", foto=lambda im: _gamma(im, 0.70),
                 nota="gamma 0.7 (aclara las sombras)"),
        variante("gamma_alta", foto=lambda im: _gamma(im, 1.40),
                 nota="gamma 1.4 (hunde las sombras)"),
        variante("saturacion_baja", foto=lambda im: _saturacion(im, 0.5),
                 nota="mitad de saturacion"),
        variante("saturacion_alta", foto=lambda im: _saturacion(im, 1.5),
                 nota="saturacion x1.5"),
        variante("balance_calido", foto=lambda im: _balance(im, 0.90, 1.00, 1.12),
                 nota="balance de blancos calido (mueve Cr, el canal de gas)"),
        variante("balance_frio", foto=lambda im: _balance(im, 1.12, 1.00, 0.90),
                 nota="balance de blancos frio"),
        variante("gris", foto=_gris3,
                 nota="sin color: prueba cuanto de CLIPSeg era semantica de color"),

        variante("ruido5", foto=lambda im: _ruido(im, 5.0),
                 nota="gaussiano sigma=5: sube ruido_base y con el los umbrales"),
        variante("jpeg40", foto=lambda im: _jpeg(im, 40),
                 nota="recompresion agresiva"),
        variante("desenfoque", foto=lambda im: cv2.GaussianBlur(im, (5, 5), 1.2),
                 nota="borra textura fina: golpea la novedad y el gradiente"),
        variante("clahe", foto=_clahe,
                 nota="ecualizacion local (lo que hace el preproceso de F0)"),
    ]
    if cuales is None:
        return V
    porNombre = {v["nombre"]: v for v in V}
    faltan = [c for c in cuales if c not in porNombre]
    if faltan:
        raise KeyError(f"Variantes desconocidas: {faltan}. "
                       f"Hay: {sorted(porNombre)}")
    return [porNombre[c] for c in cuales]


def escalar_umbrales_px(par_det, par_sam, k):
    """Reescala los umbrales que estan en PIXELES por un cambio de escala k."""
    d, s = dict(par_det), dict(par_sam)
    for clave in ("clip_fuerte", "area_min_semilla"):
        if clave in s:
            s[clave] = int(round(s[clave] * k * k))
    for clave in ("lado_min_px",):
        if clave in d:
            d[clave] = int(round(d[clave] * k))
    if "novedad_blur" in d:
        b = int(round(d["novedad_blur"] * k))
        d["novedad_blur"] = b + 1 if b % 2 == 0 else b
    return d, s


def _mapear_a_referencia(m, var, forma_ref):
    """Mascara en el marco de la variante -> marco de referencia."""
    m = var["inv_mascara"](np.asarray(m, np.uint8))
    if m.shape != tuple(forma_ref):
        m = cv2.resize(m, (forma_ref[1], forma_ref[0]),
                       interpolation=cv2.INTER_NEAREST)
    return m


def _correr_variante(var, cfg, detector, seg, masks_ref, geo, par_det, par_sam,
                     fps, fin, pivote, gsd_ref, pivote_ref_px, forma_ref,
                     dir_frames, frames_consenso, guardar_mascaras=True,
                     semilla=None, modo_semilla="random", semillas_ref=None,
                     ancla_ref=None, verbose=True):
    """Corre la Fase 1 completa sobre una edicion del video."""
    escala_v = cfg.escala * var["escala_rel"]
    k = var["escala_rel"]

    if semilla is not None:
        np.random.seed(int(semilla))

    with F0.LectorVideo(cfg.video, escala_v) as lec:
        f0 = lec[0]
        if f0 is None:
            raise RuntimeError("No se pudo leer el frame 0 del video.")
        H_s, W_s = f0.shape[:2]

    with video_transformado(var["fn"], escala_v):
        with F0.LectorVideo(cfg.video, escala_v) as lec:
            H_v, W_v = lec[0].shape[:2]
        if (H_v, W_v) != tuple(var["forma"](H_s, W_s)):
            raise RuntimeError(
                f"{var['nombre']}: la edicion devolvio {(H_v, W_v)} pero la "
                f"geometria '{var['geom']}' predice "
                f"{tuple(var['forma'](H_s, W_s))} sobre {(H_s, W_s)}.")

        px_s = pivote_ref_px[0] * (W_s / forma_ref[1])
        py_s = pivote_ref_px[1] * (H_s / forma_ref[0])
        px_v, py_v = var["punto"](px_s, py_s, H_s, W_s)
        gsd_v = gsd_ref / k

        d_v, s_v = (escalar_umbrales_px(par_det, par_sam, k)
                    if var["escalar_umbrales"] else (dict(par_det), dict(par_sam)))

        if verbose:
            print(f"  forma {H_v}x{W_v} | escala {escala_v:g} | "
                  f"gsd {gsd_v:.5f} m/px | pivote ({px_v:.0f}, {py_v:.0f})")
            if var["escalar_umbrales"] and k != 1.0:
                print(f"  umbrales px x{k:g}: clip_fuerte "
                      f"{par_sam.get('clip_fuerte')} -> {s_v.get('clip_fuerte')} | "
                      f"lado_min_px {par_det.get('lado_min_px')} -> "
                      f"{d_v.get('lado_min_px')}")

        fondo_v, ruido_v = fondo_crudo(cfg.video, pivote, escala=escala_v,
                                       rango=geo.get("ventana_tranquila"))
        det_v = Detector(detector, (H_v, W_v), (px_v, py_v), gsd_v,
                           fondo_v, ruido_v, par=d_v, verbose=verbose)

        pts_v = None
        if modo_semilla == "fijas":
            if semillas_ref is None or ancla_ref is None:
                raise ValueError("modo_semilla='fijas' necesita `semillas_ref` "
                                 "y `ancla_ref` (ver `semillas_referencia`).")
            pts_v = transportar_semillas(semillas_ref, var, forma_ref,
                                         (H_s, W_s))
            det_v = DetectorSemillaFija(det_v, pts_v)
            s_v.update(t_busqueda_ini=(ancla_ref - pivote) / fps,
                       t_busqueda_fin=(ancla_ref - pivote + 1) / fps,
                       clip_fuerte=1, siembra_mask=False)
            if verbose:
                n_pos = int((pts_v[1] == 1).sum())
                print(f"  semillas transportadas: {len(pts_v[1])} puntos "
                      f"({n_pos}+/{len(pts_v[1])-n_pos}-) | ancla forzada a "
                      f"f{ancla_ref}")
                if k != 1.0:
                    print("  AVISO: con cambio de escala el transporte pasa "
                          "por interpolacion, no es exacto.")

        masks_v, info_v = seg.segmentar(
            cfg.video, det_v, (H_v, W_v), dir_frames, pivot_frame=pivote,
            analisis_fin=fin, fps=fps, escala=escala_v, par=s_v)

    comunes = sorted(set(masks_v) & set(masks_ref))
    filas, sub, todas = [], {}, {}
    for fi in comunes:
        mv = _bin(_mapear_a_referencia(masks_v[fi], var, forma_ref))
        mr = _bin(masks_ref[fi])
        av, ar = int(mv.sum()), int(mr.sum())
        cxv, cyv = _centroide(mv)
        cxr, cyr = _centroide(mr)
        d_px = float(np.hypot(cxv - cxr, cyv - cyr)) if av and ar else np.nan
        filas.append([var["nombre"], fi, round((fi - pivote) / fps, 3), ar, av,
                      round(av / ar, 4) if ar else np.nan,
                      round(iou_mascaras(mv, mr), 4),
                      round(dice_mascaras(mv, mr), 4),
                      round(d_px, 2),
                      round(d_px * gsd_ref, 2) if not np.isnan(d_px) else np.nan])
        if fi in frames_consenso:
            sub[fi] = F0.empaquetar(mv)
        if guardar_mascaras:
            todas[fi] = F0.empaquetar(mv)

    solo_v = sorted(set(masks_v) - set(masks_ref))
    solo_r = sorted(set(masks_ref) - set(masks_v))
    meta = dict(nombre=var["nombre"], nota=var["nota"], escala=escala_v,
                forma=(H_v, W_v), gsd=gsd_v, pivote_px=(px_v, py_v),
                ruido_base=ruido_v, ancla=info_v.get("ancla"),
                nacimiento=info_v.get("nacimiento"), n_mascaras=len(masks_v),
                n_comunes=len(comunes), n_solo_variante=len(solo_v),
                n_solo_referencia=len(solo_r),
                umbrales=dict(det=d_v, sam=s_v), semilla=semilla,
                modo_semilla=modo_semilla, ancla_ref=ancla_ref,
                n_semillas=(len(pts_v[1]) if pts_v is not None else None),
                forma_ref=tuple(forma_ref))
    return filas, sub, meta, todas


COLUMNAS_CONSISTENCIA = ["variante", "frame", "t_s", "area_ref", "area_var",
                         "razon_area", "iou", "dice", "d_centroide_px",
                         "d_centroide_m"]


def clave_cache_variante(firma, nombre, mascaras=False):
    """Nombre del .pkl de una variante. `mascaras=True` -> el archivo pesado."""
    return f"consist_{firma}_{nombre}" + ("_masks" if mascaras else "")


def _resumir_variante(nombre, var, f_v, meta):
    A = np.array([[r[5], r[6], r[8]] for r in f_v], float) if f_v else \
        np.zeros((0, 3))

    def med(c, q=None):
        if not len(A) or np.isnan(A[:, c]).all():
            return np.nan
        return float(np.nanmedian(A[:, c]) if q is None
                     else np.nanpercentile(A[:, c], q))

    return dict(
        variante=nombre, estado=meta.get("estado", "ok"),
        nota=var["nota"], n=len(f_v),
        iou_mediana=med(1), iou_p10=med(1, 10),
        razon_area_mediana=med(0), razon_area_p10=med(0, 10),
        razon_area_p90=med(0, 90), d_centroide_mediana_px=med(2),
        ancla=meta.get("ancla"), nacimiento=meta.get("nacimiento"),
        n_mascaras=meta.get("n_mascaras"),
        solo_variante=meta.get("n_solo_variante"),
        solo_referencia=meta.get("n_solo_referencia"),
        ruido_base=round(float(meta.get("ruido_base", np.nan)), 3),
        minutos=round(float(meta.get("minutos", np.nan)), 1),
        error=meta.get("error", ""))


def estudio_consistencia(cfg, masks_ref, detector, seg, geo, par_det, par_sam,
                         fps, fin, pivote, gsd_ref, pivote_ref_px, forma_ref,
                         dir_frames, variantes=None, firma="", n_consenso=40,
                         usar_cache=True, guardar_mascaras=True, semilla=None,
                         seguir_si_falla=True, modo_semilla="random",
                         semillas_ref=None, ancla_ref=None, verbose=True):
    """Re-corre la Fase 1 sobre cada edicion del video y mide la dispersion."""
    if modo_semilla not in ("random", "fijas"):
        raise ValueError(f"modo_semilla '{modo_semilla}': usa 'random' o 'fijas'.")
    variantes = variantes if variantes is not None else catalogo_variantes()
    claves = sorted(masks_ref)
    idx = np.linspace(0, len(claves) - 1, min(n_consenso, len(claves)))
    frames_consenso = set(claves[int(round(i))] for i in idx)

    filas, resumen, consenso = [], [], {}
    piso_control, t_ini = None, time.time()
    for i, var in enumerate(variantes):
        nombre = var["nombre"]
        print(f"\n{'='*74}\n[{i+1}/{len(variantes)}] {nombre}  --  {var['nota']}"
              f"\n{'='*74}")
        obj = (F0.cargar_cache(clave_cache_variante(firma, nombre), cfg.dir_cache)
               if usar_cache else None)
        if obj is not None and obj.get("columnas") == COLUMNAS_CONSISTENCIA:
            f_v, sub, meta = obj["filas"], obj["sub"], obj["meta"]
            if meta.get("estado") == "error":
                print(f"  desde cache: FALLO ({meta.get('error', '')[:90]})")
            else:
                print(f"  desde cache ({len(f_v)} frames comunes, "
                      f"{meta.get('minutos', float('nan')):.1f} min ahorrados)")
        else:
            t0 = time.time()
            try:
                f_v, sub, meta, todas = _correr_variante(
                    var, cfg, detector, seg, masks_ref, geo, par_det, par_sam,
                    fps, fin, pivote, gsd_ref, pivote_ref_px, forma_ref,
                    dir_frames, frames_consenso,
                    guardar_mascaras=guardar_mascaras, semilla=semilla,
                    modo_semilla=modo_semilla, semillas_ref=semillas_ref,
                    ancla_ref=ancla_ref, verbose=verbose)
                meta["estado"] = "ok"
            except Exception as e:
                if not seguir_si_falla:
                    raise
                f_v, sub, todas = [], {}, {}
                meta = dict(nombre=nombre, estado="error",
                            error=f"{type(e).__name__}: {e}")
                print(f"\n  *** {nombre} FALLO y se anota como tal; el estudio "
                      f"sigue.\n      {type(e).__name__}: {e}\n")
            meta["minutos"] = (time.time() - t0) / 60.0
            F0.guardar_cache(dict(filas=f_v, sub=sub, meta=meta,
                                  columnas=COLUMNAS_CONSISTENCIA),
                             clave_cache_variante(firma, nombre), cfg.dir_cache)
            if todas:
                F0.guardar_cache(dict(masks=todas, forma_ref=tuple(forma_ref),
                                      variante=nombre),
                                 clave_cache_variante(firma, nombre, True),
                                 cfg.dir_cache)
            rest = len(variantes) - (i + 1)
            print(f"  {meta['minutos']:.1f} min | quedan {rest} variantes "
                  f"(~{meta['minutos']*rest:.0f} min si duran lo mismo)")
        filas += f_v
        if sub:
            consenso[nombre] = sub

        fila = _resumir_variante(nombre, var, f_v, meta)
        resumen.append(fila)

        if nombre == "identidad" and not np.isnan(fila["iou_mediana"]):
            piso_control = fila["iou_mediana"]
            _informe_control(piso_control, modo_semilla)

    ok = sum(1 for r in resumen if r["estado"] == "ok")
    print(f"\nTotal: {(time.time()-t_ini)/60:.1f} min | {ok}/{len(resumen)} "
          f"variantes con resultado")
    for r in resumen:
        if r["estado"] == "error":
            print(f"  FALLO {r['variante']}: {r['error'][:110]}")
    return filas, resumen, consenso


def _informe_control(piso, modo_semilla="random"):
    """Que significa el IoU del control `identidad`."""
    if modo_semilla == "fijas" and piso < 0.999:
        print(f"\n  *** CONTROL {piso:.4f} con semillas FIJAS. En este modo las "
              f"semillas vienen\n      dadas, asi que el azar de "
              f"`Detector.sembrar` no interviene: deberia dar\n      1.0000. "
              f"Revisa que `ancla_ref` sea el ancla real de la corrida de "
              f"referencia.\n")
        return
    if piso >= 0.99999:
        print("  control OK: identidad reproduce la corrida de produccion "
              "bit a bit.")
        return
    if piso < 0.99:
        print(f"\n  *** CONTROL FALLIDO: identidad da IoU {piso:.4f}. Eso no es "
              f"ruido de siembra:\n      el andamio esta mal y el resto del "
              f"estudio no es interpretable.\n")
        return
    print(f"\n  CONTROL = {piso:.4f}, no 1.0000. No es un fallo del andamio: es "
          f"que la\n  FASE 1 NO ES DETERMINISTA. `Detector.sembrar` elige los "
          f"{8} negativos con\n  `np.random.choice` SIN semilla (bloque D), y "
          f"con preset='hibrido' la siembra\n  es por puntos. Dos corridas de la "
          f"MISMA configuracion difieren en ~{1-piso:.2%}.\n"
          f"\n  Consecuencia para leer la tabla: {piso:.4f} es el PISO DE RUIDO. "
          f"Una variante\n  con IoU >= {piso:.4f} es indistinguible del control. "
          f"Solo lo que quede\n  claramente por debajo es efecto de la edicion.\n"
          f"\n  Para eliminarlo: `semilla=0` en estudio_consistencia (invalida "
          f"la cache) o\n  `np.random.seed(0)` en C2 antes de segmentar, y "
          f"anotarlo en la memoria.\n")


def consenso_mascaras(consenso, forma_ref, minimo=None, verbose=True):
    """Voto por mayoria entre variantes -> pseudo verdad-terreno."""
    nombres = [n for n, d in consenso.items() if d]
    frames = sorted(set.intersection(*[set(consenso[n]) for n in nombres])) \
        if nombres else []
    if not frames:
        return [], [], {}
    minimo = minimo or int(np.ceil(len(nombres) / 2.0))
    columnas = ["frame", "n_variantes", "area_consenso", "cv_area",
                "iou_media_vs_consenso", "iou_min_vs_consenso", "variante_min"]
    filas, cons = [], {}
    for fi in frames:
        ms = {n: _bin(F0.desempaquetar(consenso[n][fi], forma_ref))
              for n in nombres}
        votos = np.zeros(forma_ref, np.uint8)
        for m in ms.values():
            votos += m.astype(np.uint8)
        c = votos >= minimo
        areas = np.array([m.sum() for m in ms.values()], float)
        ious = {n: iou_mascaras(m, c) for n, m in ms.items()}
        peor = min(ious, key=ious.get)
        filas.append([fi, len(nombres), int(c.sum()),
                      round(float(areas.std() / areas.mean()), 4) if areas.mean() else np.nan,
                      round(float(np.mean(list(ious.values()))), 4),
                      round(float(ious[peor]), 4), peor])
        cons[fi] = F0.empaquetar(c)
    if verbose:
        cv = np.array([f[3] for f in filas], float)
        iu = np.array([f[4] for f in filas], float)
        print(f"[consenso] {len(nombres)} variantes, {len(frames)} frames, "
              f"voto >= {minimo}")
        print(f"  CV del conteo de pixeles : mediana {np.nanmedian(cv):.3f} "
              f"| p90 {np.nanpercentile(cv, 90):.3f}")
        print(f"  IoU medio vs consenso    : mediana {np.nanmedian(iu):.3f} "
              f"| min {np.nanmin(iu):.3f}")
    return filas, columnas, cons


FAMILIA = {
    "identidad": "control",
    "full": "resolución", "full_umbral_fijo": "resolución",
    "media_baja": "resolución",
    "rot90": "geometría", "rot180": "geometría", "rot270": "geometría",
    "espejo_h": "geometría", "espejo_v": "geometría",
    "brillo_mas": "fotometría", "brillo_menos": "fotometría",
    "contraste_alto": "fotometría", "contraste_bajo": "fotometría",
    "gamma_baja": "fotometría", "gamma_alta": "fotometría",
    "saturacion_baja": "fotometría", "saturacion_alta": "fotometría",
    "balance_calido": "fotometría", "balance_frio": "fotometría",
    "gris": "fotometría",
    "ruido5": "sensor", "jpeg40": "sensor", "desenfoque": "sensor",
    "clahe": "sensor",
}

_C_FAMILIA = {
    "control":    "#000000",
    "resolución": "#1f77b4",
    "geometría":  "#d62728",
    "fotometría": "#2ca02c",
    "sensor":     "#9467bd",
    "otras":      "#7f7f7f",
}

_ORDEN_FAMILIA = ["control", "resolución", "geometría", "fotometría",
                  "sensor", "otras"]

FIGSIZE_INFORME = (7.5, 4.3)
DPI_INFORME = 200


def familia(nombre):
    return FAMILIA.get(nombre, "otras")


def color_familia(nombre):
    return _C_FAMILIA[familia(nombre)]


def _estilo(nombre):
    """Trazo solido siempre; el control mas grueso y por encima."""
    if familia(nombre) == "control":
        return dict(color="#000000", lw=2.4, alpha=1.0, zorder=5)
    return dict(color=color_familia(nombre), lw=1.1, alpha=.80, zorder=2)


def _leyenda_familias(ax, nombres, loc="best"):
    from matplotlib.lines import Line2D
    presentes = {familia(n) for n in nombres}
    h = [Line2D([0], [0], color=_C_FAMILIA[f],
                lw=2.4 if f == "control" else 1.6, label=f)
         for f in _ORDEN_FAMILIA if f in presentes]
    ax.legend(handles=h, frameon=False, fontsize=8.5, loc=loc)


def _por_variante(filas):
    d = {}
    for r in filas:
        d.setdefault(r[0], []).append(r)
    for n in d:
        d[n].sort(key=lambda r: r[2])
    return d


def _guardar(fig, ruta):
    if ruta:
        fig.savefig(ruta, dpi=DPI_INFORME, bbox_inches="tight")
        print(f"[figura] {F0.ruta_corta(ruta)}")
    import matplotlib.pyplot as plt
    plt.show()


def grafico_conteo(filas, ruta=None):
    """Area de la mascara de cada edicion contra el tiempo, en escala log."""
    import matplotlib.pyplot as plt
    porVar = _por_variante(filas)
    fig, ax = plt.subplots(figsize=FIGSIZE_INFORME)
    for n, rs in porVar.items():
        t = [r[2] for r in rs]
        a = [r[4] if r[4] else np.nan for r in rs]
        ax.plot(t, a, **_estilo(n))
    ax.set_yscale("log")
    ax.set_xlabel("t desde la detonación (s)")
    ax.set_ylabel("área de la máscara (px$^2$)")
    ax.set_title("Conteo de píxeles")
    ax.grid(alpha=.3, which="both", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    _leyenda_familias(ax, porVar, loc="lower right")
    fig.tight_layout()
    _guardar(fig, ruta)


def grafico_razon_area(filas, ruta=None):
    """Area de cada edicion dividida por la de la corrida de referencia."""
    import matplotlib.pyplot as plt
    porVar = _por_variante(filas)
    fig, ax = plt.subplots(figsize=FIGSIZE_INFORME)
    for n, rs in porVar.items():
        t = [r[2] for r in rs]
        q = [r[5] if r[4] else np.nan for r in rs]
        ax.plot(t, q, **_estilo(n))
    ax.axhline(1.0, color="0.25", lw=1.0, ls="--", zorder=1)
    ax.set_xlabel("t desde la detonación (s)")
    ax.set_ylabel("área variante / referencia")
    ax.set_title("Error relativo (Razón de área)")
    ax.grid(alpha=.3, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    _leyenda_familias(ax, porVar, loc="upper left")
    fig.tight_layout()
    _guardar(fig, ruta)


def grafico_ranking(resumen, ruta=None, xmin=None):
    """IoU mediana de cada edicion contra la referencia, ordenado."""
    import matplotlib.pyplot as plt

    ok = [r for r in resumen if r.get("estado") == "ok"
          and r["variante"] != "identidad"
          and not np.isnan(r["iou_mediana"])]
    ok = sorted(ok, key=lambda r: r["iou_mediana"])
    mal = [r for r in resumen if r.get("estado") != "ok"]
    if not ok and not mal:
        raise RuntimeError("No hay variantes que graficar.")

    if xmin is None:
        peor = min([r["iou_mediana"] for r in ok], default=0.8)
        xmin = max(0.0, np.floor((peor - 0.05) * 20) / 20)

    nombres = [r["variante"] for r in mal] + [r["variante"] for r in ok]
    y = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(FIGSIZE_INFORME[0],
                                    max(2.6, 0.34 * len(nombres) + 1.4)))
    ax.barh(y[:len(mal)], [0] * len(mal), left=xmin, color="0.75", height=.7)
    ax.barh(y[len(mal):], [r["iou_mediana"] - xmin for r in ok], left=xmin,
            color=[color_familia(r["variante"]) for r in ok], alpha=.85,
            height=.7)
    for i, r in enumerate(mal):
        msg = str(r.get("error", "")).split(": ", 1)[-1].split(":")[0]
        ax.text(xmin + (1 - xmin) * 0.01, i,
                f"sin resultado — {msg.lower()}"[:60], fontsize=7.5,
                va="center", color="0.35")
    for i, r in enumerate(ok, start=len(mal)):
        ax.text(r["iou_mediana"] - (1 - xmin) * 0.012, i,
                f"{r['iou_mediana']:.3f}", fontsize=8, va="center", ha="right",
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=8.5)
    ax.set_xlim(xmin, 1.0)
    ax.set_ylim(-0.7, len(nombres) - 0.3)
    ax.set_xlabel("IoU mediana vs referencia")
    ax.set_title("Ranking de robustez")
    ax.grid(alpha=.3, axis="x", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _guardar(fig, ruta)


def tabla_consistencia(resumen):
    """Imprime el resumen ordenado por robustez, con el piso del control."""
    ok = [r for r in resumen if r["estado"] == "ok" and not
          np.isnan(r["iou_mediana"])]
    mal = [r for r in resumen if r not in ok]
    piso = next((r["iou_mediana"] for r in ok if r["variante"] == "identidad"),
                None)

    print(f"\n{'variante':<20}{'IoU med':>9}{'IoU p10':>9}{'area v/r':>10}"
          f"{'p10-p90':>14}{'dcent px':>10}{'ancla':>7}{'min':>7}")
    print("-" * 86)
    for r in sorted(ok, key=lambda r: -r["iou_mediana"]):
        marca = "  <- control" if r["variante"] == "identidad" else (
            "  *" if piso and r["iou_mediana"] >= piso else "")
        print(f"{r['variante']:<20}{r['iou_mediana']:>9.3f}{r['iou_p10']:>9.3f}"
              f"{r['razon_area_mediana']:>10.3f}"
              f"{r['razon_area_p10']:>7.2f}-{r['razon_area_p90']:<6.2f}"
              f"{r['d_centroide_mediana_px']:>10.1f}"
              f"{str(r['ancla']):>7}{r['minutos']:>7.1f}{marca}")
    for r in mal:
        print(f"{r['variante']:<20}{'--':>9}{'--':>9}{'--':>10}{'--':>14}"
              f"{'--':>10}{'--':>7}{r['minutos']:>7.1f}  "
              f"{r['error'][:60] or 'sin filas comunes'}")

    print("\nIoU med  : mediana del IoU contra la corrida de produccion")
    print("area v/r : mediana de pixeles(variante)/pixeles(referencia)."
          "  1.00 = mismo tamano")
    print("ancla    : frame donde cada variante decidio empezar. Si se mueve, "
          "el resto se mueve con el.")
    if piso is not None and piso < 0.99999:
        print(f"\nPISO DE RUIDO = {piso:.4f} (el control). Las marcadas con * "
              f"estan por encima:\nson indistinguibles de correr dos veces la "
              f"misma configuracion, no de la edicion.")
    print("\nAtribucion: si `ancla` se mueve, la inestabilidad nacio en "
          "CLIPSeg/novedad.\nSi el ancla es la misma y el IoU baja, es la "
          "propagacion de SAM 2.")


class DetectorSemillaFija:
    """Envuelve un `Detector` y le impone las semillas desde fuera."""

    def __init__(self, base, puntos):
        self._base = base
        self._pts = (np.asarray(puntos[0], np.float32),
                     np.asarray(puntos[1], np.int32))
        self.novedad_ancla = 0.0

    def __getattr__(self, nombre):
        return getattr(self._base, nombre)

    def sembrar(self, small, r_eq_px=None):
        if small is None:
            return None
        return self._pts


def transportar_semillas(pts, var, forma_ref, forma_var):
    """Los mismos puntos, en el marco de una edicion. (coords, labels)."""
    coords, labels = pts
    H_s, W_s = forma_var
    kx, ky = W_s / forma_ref[1], H_s / forma_ref[0]
    fuera = []
    for x, y in coords:
        xv, yv = var["punto"](float(x) * kx, float(y) * ky, H_s, W_s)
        fuera.append((xv, yv))
    return np.asarray(fuera, np.float32), np.asarray(labels, np.int32)


def _bits_variante(cfg, firma, nombre, forma_ref, masks_ref=None):
    """({frame: bits}, fuente). Prefiere las mascaras completas."""
    obj = F0.cargar_cache(clave_cache_variante(firma, nombre, True),
                          cfg.dir_cache, verbose=False)
    if obj is not None:
        return obj["masks"], "completa"
    obj = F0.cargar_cache(clave_cache_variante(firma, nombre), cfg.dir_cache,
                          verbose=False)
    if obj is not None and obj.get("sub"):
        return obj["sub"], "consenso"
    if nombre == "identidad" and masks_ref is not None:
        return {int(f): F0.empaquetar(np.asarray(m) > 0)
                for f, m in masks_ref.items()}, "referencia"
    return None, "ausente"


def inventario_mascaras(cfg, firma, nombres, forma_ref, masks_ref=None,
                        verbose=True):
    """Que hay en cache para cada variante, antes de pedir un video."""
    inv = {}
    for n in nombres:
        d, fuente = _bits_variante(cfg, firma, n, forma_ref, masks_ref)
        inv[n] = (len(d) if d else 0, fuente)
    if verbose:
        print(f"{'variante':<20}{'frames':>8}  fuente")
        print("-" * 48)
        for n, (k, f) in inv.items():
            print(f"{n:<20}{k:>8}  {f}")
        parciales = [n for n, (_, f) in inv.items() if f == "consenso"]
        ausentes = [n for n, (_, f) in inv.items() if f == "ausente"]
        if parciales:
            print(f"\n  {len(parciales)} variante(s) solo con el muestreo del "
                  f"consenso: {parciales}")
            print("  El video quedara limitado a esos frames. Para tenerlas "
                  "completas:")
            print(f"    F1.borrar_cache_variantes(CFG, firma, {parciales})")
            print("    y volver a correr C8 (solo re-corre las borradas).")
        if ausentes:
            print(f"\n  SIN CACHE: {ausentes}")
    return inv


def borrar_cache_variantes(cfg, firma, nombres, verbose=True):
    """Borra el cache de esas variantes para volver a correrlas."""
    n_borr = 0
    for n in nombres:
        for pesado in (False, True):
            r = os.path.join(cfg.dir_cache,
                             f"{clave_cache_variante(firma, n, pesado)}.pkl")
            if os.path.exists(r):
                os.remove(r)
                n_borr += 1
                if verbose:
                    print(f"  borrado {os.path.basename(r)}")
    if verbose:
        print(f"[cache] {n_borr} archivo(s). C8 volvera a correr "
              f"{len(nombres)} variante(s).")
    return n_borr


SUBCARPETAS = {
    "interno": "SAM 2 interno",
    "fijas":   "Semillas Fijas",
    "consistencia": "Consistencia",
    "gt":      "Ground truth",
}


def _repeticiones(frames, fps_fuente, fps_salida, velocidad):
    """Cuantas veces escribir cada mascara para respetar el tiempo REAL."""
    reps = []
    for k, fi in enumerate(frames):
        if k + 1 < len(frames):
            d = frames[k + 1] - fi
        else:
            d = fi - frames[k - 1] if k else 1
        dt = d / max(fps_fuente, 1e-6)
        reps.append(max(1, int(round(dt / max(velocidad, 1e-6) * fps_salida))))
    return reps


def _contorno(vis, m, color, grosor=2):
    cnts, _ = cv2.findContours(np.asarray(m, np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, grosor)


def _pintar(vis, m, color, alfa):
    if not m.any():
        return vis
    capa = vis.copy()
    capa[m] = color
    return cv2.addWeighted(capa, alfa, vis, 1 - alfa, 0)


def _texto(vis, lineas, x=15, y0=34, dy=30, escala=0.62):
    for k, (txt, color) in enumerate(lineas):
        y = y0 + k * dy
        cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    color, 2, cv2.LINE_AA)


def video_par(cfg, firma, var_a, var_b, forma_ref, ruta_salida, fps_fuente,
              masks_ref=None, etiqueta_a=None, etiqueta_b=None, alfa=0.38,
              velocidad=0.5, fps_salida=30.0, verbose=True):
    """Dos ediciones enfrentadas sobre el video original."""
    C_A, C_B, C_AMBAS = (235, 140, 40), (40, 140, 235), (150, 150, 150)
    da, fa = _bits_variante(cfg, firma, var_a, forma_ref, masks_ref)
    db, fb = _bits_variante(cfg, firma, var_b, forma_ref, masks_ref)
    if da is None or db is None:
        raise RuntimeError(f"Sin mascaras para "
                           f"{var_a if da is None else var_b} en el cache.")
    frames = sorted(set(da) & set(db))
    if not frames:
        raise RuntimeError(f"{var_a} ({fa}) y {var_b} ({fb}) no comparten "
                           f"frames. Re-corre la que este en 'consenso'.")
    ea = etiqueta_a or var_a
    eb = etiqueta_b or var_b
    if verbose:
        print(f"[video par] {ea} ({fa}, {len(da)}f) vs {eb} ({fb}, {len(db)}f)"
              f" -> {len(frames)} frames comunes")
        if min(fa, fb) == "consenso":
            print(f"   AVISO: sale un flipbook de {len(frames)} frames, no un "
                  f"video continuo.")

    H, W = forma_ref
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps_salida, W, H)
    piv = int(cfg.pivot_frame or 0)
    reps = _repeticiones(frames, fps_fuente, fps_salida, velocidad)
    if verbose:
        print(f"   {sum(reps)/fps_salida:.1f} s de video para "
              f"{(frames[-1]-frames[0])/fps_fuente:.1f} s de evento "
              f"(velocidad {velocidad:g}x)")
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for _k, fi in enumerate(frames):
            small = lec[fi]
            if small is None:
                continue
            if small.shape[:2] != (H, W):
                small = cv2.resize(small, (W, H))
            small = small[:Hv, :Wv]
            ma = (F0.desempaquetar(da[fi], forma_ref) > 0)[:Hv, :Wv]
            mb = (F0.desempaquetar(db[fi], forma_ref) > 0)[:Hv, :Wv]
            vis = _pintar(small, ma & mb, C_AMBAS, alfa * 0.55)
            vis = _pintar(vis, ma & ~mb, C_A, alfa)
            vis = _pintar(vis, mb & ~ma, C_B, alfa)
            _contorno(vis, ma, C_A)
            _contorno(vis, mb, C_B)
            aa, ab = int(ma.sum()), int(mb.sum())
            inter = int((ma & mb).sum())
            uni = int((ma | mb).sum())
            _texto(vis, [
                (f"f{fi}  t={(fi-piv)/fps_fuente:+.2f}s   IoU={inter/uni if uni else 1:.3f}",
                 (255, 255, 255)),
                (f"{ea}: {aa:,} px", C_A),
                (f"{eb}: {ab:,} px   ({ab/aa if aa else float('nan'):.3f}x)", C_B),
                (f"solo una u otra: {uni-inter:,} px "
                 f"({(uni-inter)/uni*100 if uni else 0:.1f}% de la union)",
                 (255, 255, 255)),
            ])
            for _ in range(reps[_k]):
                vw.write(vis)
    vw.release()
    if verbose:
        print(f"[video] {F0.ruta_corta(ruta_salida)}")
    return ruta_salida


def video_envolvente(cfg, firma, nombres, forma_ref, ruta_salida, fps_fuente,
                     masks_ref=None, minimo=None, alfa=0.42, velocidad=0.5,
                     fps_salida=30.0, titulo=None, verbose=True):
    """NUCLEO (todas las ediciones de acuerdo) contra ENVOLVENTE (>=1 edicion)."""
    acum, vistas, fuentes = {}, {}, {}
    n_ok = 0
    for nombre in nombres:
        d, fuente = _bits_variante(cfg, firma, nombre, forma_ref, masks_ref)
        fuentes[nombre] = (len(d) if d else 0, fuente)
        if d is None:
            continue
        n_ok += 1
        for fi, b in d.items():
            m = F0.desempaquetar(b, forma_ref) > 0
            if fi not in acum:
                acum[fi] = [F0.empaquetar(m), F0.empaquetar(m), m.astype(np.uint8)]
            else:
                u = F0.desempaquetar(acum[fi][0], forma_ref) > 0
                i_ = F0.desempaquetar(acum[fi][1], forma_ref) > 0
                acum[fi][0] = F0.empaquetar(u | m)
                acum[fi][1] = F0.empaquetar(i_ & m)
                acum[fi][2] += m.astype(np.uint8)
            vistas[fi] = vistas.get(fi, 0) + 1
        del d
        gc.collect()

    if not n_ok:
        raise RuntimeError("Ninguna variante tiene mascaras en el cache.")
    frames = sorted(f for f in acum if vistas[f] == n_ok)
    if not frames:
        raise RuntimeError(
            f"Ningun frame esta en las {n_ok} variantes a la vez. Las que "
            f"estan en 'consenso' solo aportan ~40 frames: "
            f"{[n for n, (_, f) in fuentes.items() if f == 'consenso']}")
    if verbose:
        print(f"[envolvente] {n_ok} variantes | {len(frames)} frames comunes")
        for n, (k, f) in fuentes.items():
            print(f"    {n:<20}{k:>6} f  {f}")
        if minimo:
            print(f"    nucleo relajado a >= {minimo}/{n_ok} variantes")

    H, W = forma_ref
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps_salida, W, H)
    piv = int(cfg.pivot_frame or 0)
    reps = _repeticiones(frames, fps_fuente, fps_salida, velocidad)
    serie = []
    if verbose:
        print(f"   {sum(reps)/fps_salida:.1f} s de video para "
              f"{(frames[-1]-frames[0])/fps_fuente:.1f} s de evento "
              f"(velocidad {velocidad:g}x)")
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for _k, fi in enumerate(frames):
            small = lec[fi]
            if small is None:
                continue
            if small.shape[:2] != (H, W):
                small = cv2.resize(small, (W, H))
            small = small[:Hv, :Wv]
            uni = (F0.desempaquetar(acum[fi][0], forma_ref) > 0)[:Hv, :Wv]
            nuc = ((acum[fi][2] >= minimo)[:Hv, :Wv] if minimo else
                   (F0.desempaquetar(acum[fi][1], forma_ref) > 0)[:Hv, :Wv])
            banda = uni & ~nuc
            vis = _pintar(small, nuc, (80, 210, 80), alfa)
            vis = _pintar(vis, banda, (60, 60, 230), alfa)
            _contorno(vis, uni, (60, 60, 230), 2)
            _contorno(vis, nuc, (80, 210, 80), 2)
            if masks_ref is not None and fi in masks_ref:
                _contorno(vis, np.asarray(masks_ref[fi], np.uint8)[:Hv, :Wv] > 0,
                          (255, 255, 255), 1)
            a_n, a_u = int(nuc.sum()), int(uni.sum())
            a_r = (int((np.asarray(masks_ref[fi], np.uint8)[:Hv, :Wv] > 0).sum())
                   if masks_ref is not None and fi in masks_ref else 0)
            razon = a_n / a_u if a_u else 1.0
            serie.append([fi, round((fi - piv) / fps_fuente, 3), a_n, a_u,
                          round(razon, 4), int(a_u - a_n), a_r,
                          round(a_r / a_u, 4) if a_u else np.nan])
            crit = "" if not a_u or not a_r else (
                "  (va pegada al nucleo)" if a_r <= a_n * 1.05 else
                "  (va al borde de la envolvente)" if a_r >= a_u * 0.95 else "")
            _texto(vis, [
                (f"f{fi}  t={(fi-piv)/fps_fuente:+.2f}s   "
                 f"{titulo or f'{n_ok} ediciones'}", (255, 255, 255)),
                (f"VERDE  todas las {n_ok} ediciones: {a_n:,} px",
                 (80, 210, 80)),
                (f"ROJO   solo algunas: {a_u-a_n:,} px   "
                 f"(envolvente {a_u:,} px)", (60, 60, 230)),
                (f"BLANCO corrida de trabajo (escala {cfg.escala:g}): "
                 f"{a_r:,} px{crit}", (255, 255, 255)),
                (f"acuerdo = nucleo/envolvente = {razon:.3f}", (255, 255, 255)),
            ])
            for _ in range(reps[_k]):
                vw.write(vis)
    vw.release()
    if verbose:
        r = np.array([s[4] for s in serie], float)
        print(f"[video] {F0.ruta_corta(ruta_salida)}")
        print(f"  nucleo/envolvente: mediana {np.median(r):.3f} | "
              f"min {r.min():.3f} | p10 {np.percentile(r, 10):.3f}")
        print("  1.000 seria acuerdo total. Lo que falta para 1 es el ancho "
              "de la banda,\n  y es una incertidumbre del METODO, no del "
              "penacho.")
    return ruta_salida, serie, ["frame", "t_s", "area_nucleo",
                                "area_envolvente", "acuerdo", "banda_px",
                                "area_trabajo", "trabajo_sobre_envolvente"]


def dir_experimento(cfg, clave):
    """Carpeta de un experimento: 'interno' | 'fijas' | 'consistencia'."""
    if clave not in SUBCARPETAS:
        raise KeyError(f"Experimento '{clave}'. Hay: {sorted(SUBCARPETAS)}")
    return SUBCARPETAS[clave]


def guardar_diagnosticos_consistencia(cfg, clave, filas, resumen, consenso,
                                      forma_ref, graficar=True):
    """Los CSV y las TRES figuras del estudio, en la carpeta del experimento."""
    sub = dir_experimento(cfg, clave)
    cols_r = list(resumen[0])
    F0.guardar_csv([[r[c] for c in cols_r] for r in resumen], cols_r,
                   cfg.ruta(1, "diagnosticos", sub, "consistencia_resumen.csv"))
    filas_v = []
    if filas:
        F0.guardar_csv(filas, COLUMNAS_CONSISTENCIA,
                       cfg.ruta(1, "diagnosticos", sub, "consistencia.csv"))
        filas_v, cols_v, _ = consenso_mascaras(consenso, forma_ref)
        F0.guardar_csv(filas_v, cols_v,
                       cfg.ruta(1, "diagnosticos", sub,
                                "consistencia_consenso.csv"))
        if graficar:
            r = lambda n: cfg.ruta(1, "diagnosticos", sub, n)
            grafico_conteo(filas, ruta=r("conteo_pixeles.png"))
            grafico_ranking(resumen, ruta=r("ranking_robustez.png"))
            grafico_razon_area(filas, ruta=r("razon_area.png"))
    return filas_v


GEOMETRICAS = ("rot90", "rot180", "rot270", "espejo_h", "espejo_v")


def videos_diagnostico(cfg, firma, clave, variantes, forma_ref, fps_fuente,
                       masks_ref=None, velocidad=0.5, par=("identidad", "full"),
                       verbose=True):
    """Los tres videos del experimento, con nombres que dicen que son."""
    sub = dir_experimento(cfg, clave)
    r = lambda n: cfg.ruta(1, "diagnosticos", sub, n)
    reales = [v for v in variantes if v not in GEOMETRICAS]
    estres = [v for v in variantes if v in GEOMETRICAS]
    if "identidad" in variantes and "identidad" not in estres:
        estres = ["identidad"] + estres
    salidas = {}

    if par and all(p in variantes for p in par):
        salidas["par"] = video_par(
            cfg, firma, par[0], par[1], forma_ref,
            r("01_resolucion_4K_vs_media.mp4"), fps_fuente=fps_fuente,
            masks_ref=masks_ref, velocidad=velocidad,
            etiqueta_a=f"escala {cfg.escala:g} (trabajo)",
            etiqueta_b="4K nativo", verbose=verbose)

    if len(reales) >= 2:
        _r, serie, cols = video_envolvente(
            cfg, firma, reales, forma_ref,
            r("02_envolvente_perturbaciones_reales.mp4"), fps_fuente=fps_fuente,
            masks_ref=masks_ref, minimo=None, velocidad=velocidad,
            titulo=f"{len(reales)} perturbaciones reales", verbose=verbose)
        F0.guardar_csv(serie, cols, r("envolvente_perturbaciones_reales.csv"))
        salidas["reales"] = _r

    if len(estres) >= 3:
        _r, serie, cols = video_envolvente(
            cfg, firma, estres, forma_ref,
            r("03_envolvente_giros_y_espejos.mp4"), fps_fuente=fps_fuente,
            masks_ref=masks_ref, minimo=max(2, len(estres) - 1),
            velocidad=velocidad,
            titulo=f"{len(estres)} giros/espejos (voto mayoria)",
            verbose=verbose)
        F0.guardar_csv(serie, cols, r("envolvente_giros_y_espejos.csv"))
        salidas["estres"] = _r
    elif verbose:
        print(f"[videos] solo {len(estres)} variante(s) geometrica(s): "
              f"el video 03 necesita al menos 3.")
    return salidas


def _vista_variante(cfg, firma, nombre, forma_ref, masks_ref=None):
    var = catalogo_variantes([nombre])[0]
    d, fuente = _bits_variante(cfg, firma, nombre, forma_ref, masks_ref)
    error = ""
    if d is None:
        obj = F0.cargar_cache(clave_cache_variante(firma, nombre), cfg.dir_cache,
                              verbose=False)
        error = (obj or {}).get("meta", {}).get("error", "") or "sin cache"
    return var, d, fuente, error


def _panel_variante(frame, var, bits, forma_ref, m_ref=None, alfa=0.42):
    geo = _GEOM[var["geom"]][0]
    vis = var["fn"](frame)
    m = None
    if bits is not None:
        m = geo((F0.desempaquetar(bits, forma_ref) > 0).astype(np.uint8)) > 0
        vis = _pintar(vis, m, (40, 140, 235), alfa)
        _contorno(vis, m, (40, 140, 235), 2)
    if m_ref is not None:
        _contorno(vis, geo(m_ref.astype(np.uint8)) > 0, (255, 255, 255), 1)
    return vis, m


def video_variante(cfg, firma, nombre, forma_ref, ruta_salida, fps_fuente,
                   masks_ref, alfa=0.42, velocidad=0.5, fps_salida=30.0,
                   verbose=True):
    """El video editado tal como lo vio el algoritmo, con su mascara y la referencia."""
    var, d, fuente, error = _vista_variante(cfg, firma, nombre, forma_ref,
                                            masks_ref)
    frames = sorted(set(d) & set(masks_ref)) if d else sorted(masks_ref)
    H, W = var["forma"](*forma_ref)
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps_salida, W, H)
    piv = int(cfg.pivot_frame or 0)
    reps = _repeticiones(frames, fps_fuente, fps_salida, velocidad)
    iou = []
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for k, fi in enumerate(frames):
            fr = lec[fi]
            if fr is None:
                continue
            if fr.shape[:2] != tuple(forma_ref):
                fr = cv2.resize(fr, (forma_ref[1], forma_ref[0]))
            m_ref = np.asarray(masks_ref[fi]) > 0
            vis, m = _panel_variante(fr, var, d[fi] if d else None, forma_ref,
                                     m_ref, alfa)
            lineas = [(f"{nombre}: {var['nota']}", (255, 255, 255)),
                      (f"f{fi}  t={(fi - piv) / fps_fuente:+.2f}s", (255, 255, 255))]
            if m is not None:
                a_v, a_r = int(m.sum()), int(m_ref.sum())
                j = iou_mascaras(_GEOM[var["geom"]][0](m_ref.astype(np.uint8)), m)
                iou.append(j)
                lineas += [(f"variante: {a_v:,} px   ({a_v / a_r if a_r else float('nan'):.2f}x)",
                            (40, 140, 235)),
                           (f"referencia: {a_r:,} px   IoU={j:.3f}", (255, 255, 255))]
            else:
                lineas.append((f"sin mascara: {error}"[:90], (60, 60, 230)))
            _texto(vis, lineas)
            vis = np.ascontiguousarray(vis[:Hv, :Wv])
            for _ in range(reps[k]):
                vw.write(vis)
    vw.release()
    if verbose:
        txt = f"IoU mediana {np.median(iou):.3f}" if iou else "sin mascaras"
        print(f"  {nombre:<18}{fuente:<12}{len(frames):>5} f   {txt}")
    return ruta_salida


def videos_variantes(cfg, firma, nombres, forma_ref, fps_fuente, masks_ref,
                     velocidad=0.5, carpeta="variantes", verbose=True):
    """Un video por variante en diagnosticos/Consistencia/<carpeta>/."""
    sub = dir_experimento(cfg, "consistencia")
    salidas = {}
    if verbose:
        print(f"{'variante':<20}{'fuente':<12}{'frames':>7}")
    for n in nombres:
        salidas[n] = video_variante(
            cfg, firma, n, forma_ref,
            cfg.ruta(1, "diagnosticos", sub, carpeta, f"{n}.mp4"),
            fps_fuente, masks_ref, velocidad=velocidad, verbose=verbose)
        gc.collect()
    if verbose:
        print(f"[videos] {F0.ruta_corta(os.path.dirname(next(iter(salidas.values()))))}")
    return salidas


def grilla_variantes(cfg, firma, nombres, forma_ref, tiempos_s, fps_fuente,
                     masks_ref, ruta=None, ancho_panel=3.2):
    """Filas = variantes, columnas = instantes. Frame editado + mascara + referencia."""
    import matplotlib.pyplot as plt

    piv = int(cfg.pivot_frame or 0)
    disp = np.array(sorted(masks_ref))
    frames = [int(disp[np.argmin(np.abs(disp - (piv + t * fps_fuente)))])
              for t in tiempos_s]
    fig, axs = plt.subplots(len(nombres), len(frames), squeeze=False,
                            figsize=(ancho_panel * len(frames),
                                     ancho_panel * 0.62 * len(nombres)))
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        crudos = {}
        for fi in frames:
            fr = lec[fi]
            if fr.shape[:2] != tuple(forma_ref):
                fr = cv2.resize(fr, (forma_ref[1], forma_ref[0]))
            crudos[fi] = fr
    for i, n in enumerate(nombres):
        var, d, _, error = _vista_variante(cfg, firma, n, forma_ref, masks_ref)
        for j, fi in enumerate(frames):
            ax = axs[i, j]
            m_ref = np.asarray(masks_ref[fi]) > 0
            bits = d.get(fi) if d else None
            vis, m = _panel_variante(crudos[fi], var, bits, forma_ref, m_ref)
            ax.imshow(vis[:, :, ::-1])
            ax.set_xticks([]); ax.set_yticks([])
            t_txt = f"t = {(fi - piv) / fps_fuente:.1f} s"
            if m is not None:
                g = _GEOM[var["geom"]][0](m_ref.astype(np.uint8))
                ax.set_title(f"{t_txt}   IoU {iou_mascaras(g, m):.2f}", fontsize=8)
            else:
                ax.set_title(f"{t_txt}   sin máscara", fontsize=8, color="C3")
            if j == 0:
                ax.set_ylabel(n, fontsize=9)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=200, bbox_inches="tight")
        print(f"[grilla] {F0.ruta_corta(ruta)}")
    plt.show()
    return fig


def _dir_gt(cfg):
    sello = F0.sello_video(cfg.video_nombre) or os.path.splitext(cfg.video_nombre)[0]
    return cfg._dato(f"anotaciones_{sello}.json")


def poligonos_a_mascara(polys, forma):
    """Poligonos [(x,y), ...] en coords de TRABAJO -> mascara uint8 {0,1}."""
    m = np.zeros(forma, np.uint8)
    conts = [np.asarray(p, np.float32).round().astype(np.int32)
             for p in polys if len(p) >= 3]
    if conts:
        cv2.fillPoly(m, conts, 1)
    return m


def cargar_anotaciones(cfg, ruta=None):
    """El JSON de anotaciones, o una estructura vacia si no existe."""
    import json
    r = ruta or _dir_gt(cfg)
    if os.path.exists(r):
        with open(r, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return dict(video=os.path.basename(cfg.video), escala=cfg.escala,
                pivot_frame=int(cfg.pivot_frame or 0), frames={})


def guardar_anotaciones(ann, cfg, ruta=None, verbose=True):
    import json
    r = ruta or _dir_gt(cfg)
    os.makedirs(os.path.dirname(r), exist_ok=True)
    with open(r, "w", encoding="utf-8") as fh:
        json.dump(ann, fh, ensure_ascii=False, indent=1)
    if verbose:
        print(f"[gt] {len(ann['frames'])} frame(s) -> {F0.ruta_corta(r)}")
    return r


def mascaras_ground_truth(ann, forma):
    """{frame: mascara} desde los poligonos guardados."""
    return {int(f): poligonos_a_mascara(polys, forma)
            for f, polys in ann.get("frames", {}).items() if polys}


def anotar_columna(cfg, frames, forma, fps=30.0, ruta=None,
                   ancho_ventana=1500, zoom_defecto=3.0, zoom_max=12.0,
                   verbose=True):
    """Trazado manual de la columna de gas, poligono a poligono."""
    ann = cargar_anotaciones(cfg, ruta)
    ann.setdefault("frames", {})
    orden = list(frames)
    H, W = forma
    fit = min(1.0, ancho_ventana / W)
    Wv, Hv = int(W * fit), int(H * fit)
    WIN = "Anotacion manual - columna de gas"
    st = dict(i=0, curso=[], salir=False, zoom=1.0, cx=W / 2.0, cy=H / 2.0)

    def _vista():
        """(x0, y0, ancho, alto) de la region visible, en coords de trabajo."""
        z = st["zoom"]
        vw, vh = W / z, H / z
        x0 = float(np.clip(st["cx"] - vw / 2, 0, max(0.0, W - vw)))
        y0 = float(np.clip(st["cy"] - vh / 2, 0, max(0.0, H - vh)))
        return x0, y0, vw, vh

    def _a_trabajo(xw, yw):
        x0, y0, vw, vh = _vista()
        return x0 + xw * vw / Wv, y0 + yw * vh / Hv

    def _a_ventana(x, y):
        x0, y0, vw, vh = _vista()
        return (x - x0) * Wv / vw, (y - y0) * Hv / vh

    def _recorte(img, interp):
        x0, y0, vw, vh = _vista()
        r = img[int(y0):int(np.ceil(y0 + vh)), int(x0):int(np.ceil(x0 + vw))]
        if r.size == 0:
            r = img
        return cv2.resize(r, (Wv, Hv), interpolation=interp)

    def _polys(fi):
        return ann["frames"].setdefault(str(int(fi)), [])

    def _cerrar():
        if len(st["curso"]) >= 3:
            _polys(orden[st["i"]]).append([[float(a), float(b)]
                                           for a, b in st["curso"]])
        st["curso"] = []

    def _cb(ev, x, y, flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["curso"].append(_a_trabajo(x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            st["cx"], st["cy"] = _a_trabajo(x, y)
            st["zoom"] = (zoom_defecto if st["zoom"] <= 1.0
                          else min(zoom_max, st["zoom"] * 2.0))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, Wv, Hv)
    cv2.setMouseCallback(WIN, _cb)
    lec = F0.LectorVideo(cfg.video, cfg.escala)
    piv = int(cfg.pivot_frame or 0)
    if verbose:
        print(f"[gt] {len(orden)} frames | ventana {Wv}x{Hv} | "
              f"1 px de ventana = {1/fit:.1f} px de trabajo sin zoom")
        print("     izq=vertice  DER=ZOOM  c=cerrar poligono  +/-=zoom  "
              "f=cuadro completo")
        print("     z=deshacer  x=borrar poligono  r=reiniciar")
        print("     ESPACIO=siguiente  a=anterior  g=guardar  q=salir")
    try:
        while not st["salir"]:
            fi = orden[st["i"]]
            base = lec[fi]
            if base is None:
                st["i"] = (st["i"] + 1) % len(orden)
                continue
            if base.shape[:2] != (H, W):
                base = cv2.resize(base, (W, H))

            z = st["zoom"]
            vis = _recorte(base, cv2.INTER_NEAREST if z > 1.0
                           else cv2.INTER_AREA)

            hechos = _polys(fi)
            area = 0
            if hechos:
                m = poligonos_a_mascara(hechos, (H, W))
                area = int(m.sum())
                mv = _recorte(m * 255, cv2.INTER_NEAREST) > 0
                vis = _pintar(vis, mv, (90, 220, 90), 0.30)
                _contorno(vis, mv, (60, 200, 60), 2)
            if st["curso"]:
                pts = np.asarray([_a_ventana(x, y) for x, y in st["curso"]],
                                 np.int32)
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
                for x, y in pts:
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1)

            _texto(vis, [
                (f"[{st['i']+1}/{len(orden)}]  f{fi}  t={(fi-piv)/fps:+.2f}s",
                 (255, 255, 255)),
                (f"{len(hechos)} poligono(s) | {area:,} px "
                 f"({area/(H*W)*100:.2f}% del cuadro)", (90, 220, 90)),
                (f"vertices en curso: {len(st['curso'])}", (0, 255, 255)),
                (f"zoom x{z:.1f}" + ("  (der=acercar, f=cuadro completo)"
                                     if z > 1.0 else "  (der=acercar)"),
                 (200, 200, 200)),
            ])
            cv2.imshow(WIN, vis)

            k = cv2.waitKey(20) & 0xFF
            if k in (ord(" "), ord("n"), ord("d")):
                _cerrar()
                guardar_anotaciones(ann, cfg, ruta, verbose=False)
                st["i"] = (st["i"] + 1) % len(orden)
            elif k in (ord("a"), ord("p")):
                _cerrar()
                st["i"] = (st["i"] - 1) % len(orden)
            elif k == ord("c"):
                _cerrar()
            elif k in (ord("+"), ord("=")):
                st["zoom"] = min(zoom_max, st["zoom"] * 1.5)
            elif k in (ord("-"), ord("_")):
                st["zoom"] = max(1.0, st["zoom"] / 1.5)
            elif k == ord("f"):
                st["zoom"], st["cx"], st["cy"] = 1.0, W / 2.0, H / 2.0
            elif k == ord("z"):
                if st["curso"]:
                    st["curso"].pop()
                elif hechos:
                    st["curso"] = [tuple(v) for v in hechos.pop()]
            elif k == ord("x"):
                if hechos:
                    hechos.pop()
            elif k == ord("r"):
                st["curso"] = []
                ann["frames"][str(int(fi))] = []
            elif k == ord("g"):
                _cerrar()
                guardar_anotaciones(ann, cfg, ruta)
            elif k in (ord("q"), 27):
                _cerrar()
                st["salir"] = True
    finally:
        lec.close()
        cv2.destroyWindow(WIN)
        for _ in range(4):
            cv2.waitKey(1)

    ann["frames"] = {f: p for f, p in ann["frames"].items() if p}
    guardar_anotaciones(ann, cfg, ruta)
    if verbose:
        print(f"[gt] {len(ann['frames'])} de {len(orden)} frames anotados.")
        faltan = [f for f in orden if str(int(f)) not in ann["frames"]]
        if faltan:
            print(f"     sin anotar: {faltan}")
    return ann


COLUMNAS_EXITO = ["frame", "t_s", "frame_modelo", "area_gt", "area_modelo",
                  "razon_area", "error_area_pct", "px_de_mas", "px_de_menos",
                  "iou"]


def exito_por_area(ann, masks, forma, fps, pivote, tol=2, verbose=True):
    """El modelo contra el trazado manual, medido en pixeles."""
    gts = mascaras_ground_truth(ann, forma)
    if not gts:
        raise RuntimeError("No hay anotaciones. Corre `anotar_columna` primero.")
    filas = []
    for fi in sorted(gts):
        fm = fi if fi in masks else (
            min(masks, key=lambda k: abs(k - fi)) if masks else None)
        if fm is None or abs(fm - fi) > tol:
            continue
        gt = _bin(gts[fi])
        mo = _bin(F0.normalizar_mascara(masks[fm], forma))
        inter = int((gt & mo).sum())
        a_g, a_m = int(gt.sum()), int(mo.sum())
        uni = a_g + a_m - inter
        filas.append([fi, round((fi - pivote) / fps, 3), fm, a_g, a_m,
                      round(a_m / a_g, 4) if a_g else np.nan,
                      round((a_m - a_g) / a_g * 100, 2) if a_g else np.nan,
                      int(a_m - inter), int(a_g - inter),
                      round(inter / uni, 4) if uni else 1.0])
    if not filas:
        raise RuntimeError(f"Ningun frame anotado tiene mascara del modelo a "
                           f"menos de {tol} frames.")
    if verbose:
        A = np.array([[f[5], f[6], f[9]] for f in filas], float)
        print(f"\n{'frame':>7}{'t (s)':>9}{'area gt':>11}{'area mod':>11}"
              f"{'razon':>8}{'err %':>9}{'IoU':>8}")
        print("-" * 63)
        for f in filas:
            print(f"{f[0]:>7}{f[1]:>9.2f}{f[3]:>11,}{f[4]:>11,}"
                  f"{f[5]:>8.3f}{f[6]:>+9.1f}{f[9]:>8.3f}")
        print("-" * 63)
        print(f"\n  razon de area   mediana {np.nanmedian(A[:,0]):.3f}  "
              f"[{np.nanmin(A[:,0]):.3f} - {np.nanmax(A[:,0]):.3f}]")
        print(f"  error de area   mediana {np.nanmedian(A[:,1]):+.1f} %")
        print(f"  IoU             mediana {np.nanmedian(A[:,2]):.3f}  "
              f"| p10 {np.nanpercentile(A[:,2],10):.3f}")
        r, i_ = np.nanmedian(A[:, 0]), np.nanmedian(A[:, 2])
        if r > 1.05:
            print(f"  -> el modelo marca {(r-1)*100:.0f}% mas area que el "
                  f"trazado.")
        elif r < 0.95:
            print(f"  -> el modelo marca {(1-r)*100:.0f}% menos area que el "
                  f"trazado.")
        else:
            print("  -> el tamano coincide con el trazado.")
        if abs(r - 1) < 0.05 and i_ < 0.85:
            print("     Pero el IoU es bajo: mismo conteo de pixeles en otro "
                  "sitio, no la misma mascara.")
    return filas, COLUMNAS_EXITO


def grafico_area_vs_tiempo(ann, masks, forma, fps, pivote, ruta=None,
                           tol=2, verbose=True):
    """Area detectada contra area trazada a mano, en el tiempo."""
    import matplotlib.pyplot as plt

    C_SAM, C_MAN = "#1f77b4", "#2ca02c"
    gts = mascaras_ground_truth(ann, forma)
    if not gts:
        raise RuntimeError("No hay anotaciones. Corre `anotar_columna` primero.")

    fs_mo = sorted(masks)
    t_mo = [(f - pivote) / fps for f in fs_mo]
    a_mo = [int((_bin(F0.normalizar_mascara(masks[f], forma))).sum())
            for f in fs_mo]

    t_gt, a_gt, a_par, fs_gt = [], [], [], []
    for fi in sorted(gts):
        fm = fi if fi in masks else (
            min(masks, key=lambda k: abs(k - fi)) if masks else None)
        if fm is None or abs(fm - fi) > tol:
            continue
        fs_gt.append(fi)
        t_gt.append((fi - pivote) / fps)
        a_gt.append(int(_bin(gts[fi]).sum()))
        a_par.append(int((_bin(F0.normalizar_mascara(masks[fm], forma))).sum()))
    if not t_gt:
        raise RuntimeError(f"Ningun frame anotado tiene mascara del modelo a "
                           f"menos de {tol} frames.")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for t, ag, am in zip(t_gt, a_gt, a_par):
        ax.plot([t, t], [ag, am], color="0.65", lw=1, zorder=1)
    ax.plot(t_mo, a_mo, "-", color=C_SAM, lw=2, zorder=2,
            label=f"SAM 2  ({len(fs_mo)} frames)")
    ax.plot(t_gt, a_gt, "--o", color=C_MAN, lw=1.4, ms=8, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2,
            label=f"Anotacion manual  ({len(t_gt)} frames)")

    ax.set_xlabel("t desde la detonacion (s)")
    ax.set_ylabel("area de la mascara (px)")
    ax.set_title("Area de la columna: SAM 2 contra el trazado manual")
    ax.grid(alpha=.3, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    razon = np.array(a_par, float) / np.maximum(np.array(a_gt, float), 1)
    ax.annotate(f"area SAM 2 / manual:  mediana {np.median(razon):.2f}"
                f"   [{razon.min():.2f} - {razon.max():.2f}]",
                xy=(0.99, 0.03), xycoords="axes fraction", ha="right",
                fontsize=9, color="0.35")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        print(f"[figura] {F0.ruta_corta(ruta)}")
    plt.show()
    if verbose:
        print(f"  area SAM 2 / manual: mediana {np.median(razon):.3f} "
              f"| min {razon.min():.3f} | max {razon.max():.3f}")
        sub = int((razon < 0.95).sum())
        sob = int((razon > 1.05).sum())
        print(f"  de {len(razon)} frames anotados: {sob} con el modelo por "
              f"encima, {sub} por debajo, {len(razon)-sob-sub} dentro de +-5%")
    return fs_gt, a_gt, a_par


_C_SAM_BGR = (180, 119, 31)
_C_MAN_BGR = (44, 160, 44)


def figura_comparacion_mascaras(cfg, ann, masks, forma, fps, pivote, ruta=None,
                                tol=2, n_max=12, recortar=True, margen=0.18,
                                grosor=2, verbose=True):
    """Los frames anotados, con las DOS fronteras encima del frame real."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gts = mascaras_ground_truth(ann, forma)
    if not gts:
        raise RuntimeError("No hay anotaciones. Corre `anotar_columna` primero.")
    H, W = forma

    pares = []
    for fi in sorted(gts):
        fm = fi if fi in masks else (
            min(masks, key=lambda k: abs(k - fi)) if masks else None)
        if fm is not None and abs(fm - fi) <= tol:
            pares.append((fi, fm))
    if not pares:
        raise RuntimeError(f"Ningun frame anotado tiene mascara del modelo a "
                           f"menos de {tol} frames.")
    if len(pares) > n_max:
        idx = np.linspace(0, len(pares) - 1, n_max).round().astype(int)
        pares = [pares[i] for i in np.unique(idx)]

    n = len(pares)
    cols = min(4, n)
    filas = int(np.ceil(n / cols))
    fig, ax = plt.subplots(filas, cols, figsize=(4.6 * cols, 3.0 * filas),
                           squeeze=False)

    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for k, (fi, fm) in enumerate(pares):
            a = ax[k // cols][k % cols]
            base = lec[fi]
            if base is None:
                a.axis("off")
                continue
            if base.shape[:2] != (H, W):
                base = cv2.resize(base, (W, H))
            gt = _bin(gts[fi])
            mo = _bin(F0.normalizar_mascara(masks[fm], forma))

            vis = base.copy()
            _contorno(vis, gt, _C_MAN_BGR, grosor)
            _contorno(vis, mo, _C_SAM_BGR, grosor)

            if recortar:
                ys, xs = np.where(gt | mo)
                if len(xs):
                    mx = int(margen * max(np.ptp(xs), np.ptp(ys)) + 12)
                    x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + mx + 1)
                    y0, y1 = max(0, ys.min() - mx), min(H, ys.max() + mx + 1)
                    vis = vis[y0:y1, x0:x1]

            inter = int((gt & mo).sum())
            a_g, a_m = int(gt.sum()), int(mo.sum())
            uni = a_g + a_m - inter
            a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            a.set_title(f"f{fi}   t={(fi-pivote)/fps:+.1f}s   "
                        f"IoU={inter/uni if uni else 1:.3f}\n"
                        f"faltan {a_g-inter:,} px   sobran {a_m-inter:,} px",
                        fontsize=9)
            a.axis("off")

    for k in range(n, filas * cols):
        ax[k // cols][k % cols].axis("off")
    fig.legend(handles=[
        Line2D([0], [0], color=np.array(_C_MAN_BGR[::-1]) / 255, lw=2.5,
               label="trazado manual"),
        Line2D([0], [0], color=np.array(_C_SAM_BGR[::-1]) / 255, lw=2.5,
               label="SAM 2")],
        loc="upper center", ncol=2, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    if ruta:
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        print(f"[figura] {F0.ruta_corta(ruta)}")
    plt.show()
    if verbose:
        print(f"  {n} frames comparados"
              + (f" (de {len(gts)} anotados)" if n < len(gts) else ""))
    return [p[0] for p in pares]
