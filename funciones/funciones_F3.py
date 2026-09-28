"""Fase 3: viento por flujo optico sobre la mascara de la Fase 1, georreferenciado sobre el
plano de adveccion y contrastado con estaciones meteorologicas.
"""

import json
import math
import os
import re

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

import funciones_F0 as F0

FPS_DEF = 29.97002997002997


def media_circular_deg(angulos_deg):
    a = np.radians(angulos_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def media_circular_pond(angulos_deg, pesos=None):
    """Media circular ponderada (por velocidad, normalmente). En [0, 360)."""
    a = np.radians(np.asarray(angulos_deg, float))
    w = np.ones_like(a) if pesos is None else np.asarray(pesos, float)
    ok = np.isfinite(a) & np.isfinite(w)
    return math.degrees(math.atan2(np.sum(w[ok] * np.sin(a[ok])),
                                   np.sum(w[ok] * np.cos(a[ok])))) % 360.0


def dispersion_circular_deg(angulos_deg, pesos=None):
    """Desviacion circular en grados. 0 = todos iguales, ~81 = uniforme."""
    a = np.radians(np.asarray(angulos_deg, float))
    w = np.ones_like(a) if pesos is None else np.asarray(pesos, float)
    ok = np.isfinite(a) & np.isfinite(w)
    if not ok.any():
        return float("nan")
    Rb = math.hypot(np.sum(w[ok] * np.sin(a[ok])),
                    np.sum(w[ok] * np.cos(a[ok]))) / np.sum(w[ok])
    return math.degrees(math.sqrt(max(-2.0 * math.log(max(Rb, 1e-12)), 0.0)))


def suavizado_circular_deg(angulos_deg, ventana):
    a = np.radians(angulos_deg)
    s = uniform_filter1d(np.sin(a), size=ventana, mode="nearest")
    c = uniform_filter1d(np.cos(a), size=ventana, mode="nearest")
    return np.degrees(np.arctan2(s, c))


def direccion_dominante(fx, fy):
    """Eje mayor del campo de flujo por PCA, orientado con el flujo medio."""
    fmean = np.array([np.mean(fx), np.mean(fy)])
    V = np.stack([fx - fmean[0], fy - fmean[1]], axis=1)
    cov = (V.T @ V) / max(len(fx) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    eje = vecs[:, int(np.argmax(vals))]
    if np.dot(eje, fmean) < 0:
        eje = -eje
    return eje


N_BINS_DIR = 72
N_MUESTRA_DIR = 200


def ajustar_von_mises(fx, fy, nbins=N_BINS_DIR, n_muestra=N_MUESTRA_DIR):
    """Ajuste por maxima verosimilitud de una von Mises a las DIRECCIONES del campo de
    flujo de UN par de frames.
    """
    mod = np.hypot(fx, fy)
    ok = mod > 0
    if int(ok.sum()) < 10:
        return (np.nan, np.nan, np.nan, np.zeros(nbins),
                np.full(n_muestra, np.nan))
    a = np.arctan2(fy[ok], fx[ok])
    C, S = float(np.cos(a).mean()), float(np.sin(a).mean())
    mu = float(np.degrees(np.arctan2(S, C)))
    Rbar = float(min(np.hypot(C, S), 1.0 - 1e-12))
    kappa = float(Rbar * (2.0 - Rbar ** 2) / (1.0 - Rbar ** 2))
    hist, _ = np.histogram(a, bins=nbins, range=(-np.pi, np.pi), density=True)
    rng = np.random.default_rng(0)
    muestra = a[rng.integers(0, a.size, n_muestra)].astype(np.float32)
    return mu, kappa, Rbar, hist, muestra


def metricas_frame(g, m, radio_anillo=15):
    """Textura y distancia al borde de UNA mascara. Base del criterio de t_fin."""
    mk = (m > 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    k = np.ones((2 * radio_anillo + 1, 2 * radio_anillo + 1), np.uint8)
    anillo = (cv2.dilate(mk.astype(np.uint8), k) > 0) & (~mk)
    gi = g[mk].astype(np.float32)
    gr = g[anillo].astype(np.float32)
    mi, mr = mag[mk], mag[anillo]
    if mr.size:
        umbral = float(np.median(mr))
        textura = float((mi > umbral).mean())
        tex_ring = float(mr.mean())
        cnr = abs(gi.mean() - gr.mean()) / np.sqrt(0.5 * (gi.var() + gr.var()) + 1e-9)
    else:
        textura = tex_ring = cnr = np.nan
    hh, ww = g.shape
    yy, xx = np.nonzero(mk)
    db = np.minimum(np.minimum(xx, ww - 1 - xx),
                    np.minimum(yy, hh - 1 - yy)).astype(float)
    return dict(area=float(mk.sum()),
                textura=textura,
                tex_in=float(mi.mean()), tex_ring=tex_ring,
                cnr=float(cnr), std_in=float(gi.std()),
                dist_borde=float(np.percentile(db, 5)),
                frac_borde=float((db < 0.04 * min(hh, ww)).mean()),
                recorte=float((db < 1).mean()))


def estimar_viento(ruta_video, masks, fps, escala=0.5, gsd_m_px=None,
                   ventana_suave=15, ventana_dir=9, max_gap=3, min_px=200,
                   etiqueta="SAM2", trayectoria="flujo_acumulado",
                   radio_anillo=15):
    """Flujo optico sobre pares de frames CON mascara."""
    fkeys = sorted(masks)
    dx, dy, proj, frames_ok = [], [], [], []
    vmf_mu, vmf_kappa, vmf_rbar, vmf_hist, vmf_ang = [], [], [], [], []
    met = {k: [] for k in ("area", "textura", "tex_in", "tex_ring", "cnr",
                           "std_in", "dist_borde", "frac_borde", "recorte")}
    n_gap = n_pocos = n_lectura = 0
    forma_trabajo = None

    with F0.LectorVideo(ruta_video, escala) as lec:
        def gris(fi):
            im = lec[fi]
            return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        print(f"Flujo optico ({etiqueta})...")
        for k in range(1, len(fkeys)):
            fi, fi_prev = fkeys[k], fkeys[k - 1]
            salto = fi - fi_prev
            if salto > max_gap:
                n_gap += 1
                continue
            g, gp = gris(fi), gris(fi_prev)
            if g is None or gp is None:
                n_lectura += 1
                continue
            forma_trabajo = g.shape
            m = F0.normalizar_mascara(masks[fi], g.shape)
            mp = F0.normalizar_mascara(masks[fi_prev], g.shape)
            comun = cv2.bitwise_and(m, mp)
            if int(np.sum(comun > 0)) <= min_px:
                n_pocos += 1
                continue
            flow = cv2.calcOpticalFlowFarneback(gp, g, None, pyr_scale=0.5,
                                                levels=3, winsize=21, iterations=3,
                                                poly_n=7, poly_sigma=1.5, flags=0)
            fx = flow[..., 0][comun > 0]
            fy = flow[..., 1][comun > 0]
            eje = direccion_dominante(fx, fy)
            p = float(np.median(fx * eje[0] + fy * eje[1])) / salto
            dx.append(p * eje[0]); dy.append(p * eje[1])
            proj.append(p); frames_ok.append(fi)
            mu_v, kap_v, rb_v, h_v, a_v = ajustar_von_mises(fx, fy)
            vmf_mu.append(mu_v); vmf_kappa.append(kap_v)
            vmf_rbar.append(rb_v); vmf_hist.append(h_v); vmf_ang.append(a_v)
            mf = metricas_frame(g, m, radio_anillo=radio_anillo)
            for k, val in mf.items():
                met[k].append(val)

    descartes = dict(gap_grande=n_gap, pocos_px=n_pocos, lectura=n_lectura)
    if not proj:
        print(f"  Sin datos. Descartes: {descartes}")
        return None

    dx, dy = np.array(dx), np.array(dy)
    proj, frames_ok = np.array(proj), np.array(frames_ok)
    ang_raw = np.degrees(np.arctan2(dy, dx))
    ang_dir = suavizado_circular_deg(ang_raw, min(ventana_dir, len(ang_raw)))
    vel_raw = np.abs(proj) / escala * fps
    win = min(ventana_suave, len(vel_raw))
    vel = uniform_filter1d(vel_raw, size=win, mode="nearest")
    ang = suavizado_circular_deg(ang_dir, win)
    vel_ms = vel * gsd_m_px if gsd_m_px else None

    if trayectoria == "centroides":
        cx, cy = trayectoria_centroides(masks, frames_ok, forma=forma_trabajo)
    else:
        cx = np.cumsum(vel / fps * np.cos(np.radians(ang)))
        cy = np.cumsum(vel / fps * np.sin(np.radians(ang)))
        cx, cy = cx - cx[0], cy - cy[0]

    res = dict(frames=frames_ok, speed_px_s=vel, speed_m_s=vel_ms,
               angle_deg=ang, cx=cx, cy=cy, descartes=descartes,
               trayectoria=trayectoria, gsd_m_px=gsd_m_px, fps=fps,
               forma=forma_trabajo,
               angle_pca_deg=ang_raw,
               vmf_mu_deg=np.array(vmf_mu, float),
               vmf_kappa=np.array(vmf_kappa, float),
               vmf_rbar=np.array(vmf_rbar, float),
               vmf_hist=np.array(vmf_hist, float),
               vmf_muestra=np.array(vmf_ang, float),
               **{k: np.array(v, float) for k, v in met.items()})
    _imprimir_resumen(res, etiqueta)
    return res


def centroide_mascaras(masks, frames, forma=None):
    """Centroide de la mascara por frame, en pixeles de TRABAJO (absolutos)."""
    u, v = [], []
    for f in frames:
        m = masks[int(f)]
        m = F0.normalizar_mascara(m, forma) if forma is not None else np.asarray(m)
        ys, xs = np.where(m > 0)
        u.append(xs.mean() if len(xs) else np.nan)
        v.append(ys.mean() if len(ys) else np.nan)
    u, v = np.array(u, float), np.array(v, float)
    n = int(np.isnan(u).sum())
    if n:
        i = np.arange(len(u)); ok = ~np.isnan(u)
        u = np.interp(i, i[ok], u[ok]); v = np.interp(i, i[ok], v[ok])
        print(f"  [aviso] {n} mascaras vacias, centroide interpolado")
    return u, v


def trayectoria_centroides(masks, frames, forma=None):
    """Centroide de la mascara relativo al primer frame."""
    u, v = centroide_mascaras(masks, frames, forma=forma)
    return u - u[0], v - v[0]


def _imprimir_resumen(res, etiqueta):
    v, g = res["speed_px_s"], res["gsd_m_px"]
    print(f"\n{'-'*58}\n  {etiqueta}\n{'-'*58}")
    print(f"  Pares validos   : {len(res['frames'])}")
    print(f"  Descartados     : {res['descartes']}")
    linea = f"  Velocidad media : {v.mean():.1f} px/s"
    if g:
        linea += f"  ->  {v.mean()*g:.2f} m/s  ({v.mean()*g*3.6:.1f} km/h)"
    print(linea)
    print(f"  Velocidad maxima: {v.max():.1f} px/s")
    print(f"  Direccion media : {media_circular_deg(res['angle_deg']):.1f} deg "
          f"(0=E, 90=N en imagen)")
    print(f"  Trayectoria     : {res['trayectoria']}")
    print("-" * 58)


def guardar_serie(res, ruta_csv, pivot_frame=0):
    filas = []
    for i, f in enumerate(res["frames"]):
        filas.append([int(f), f"{(f - pivot_frame)/res['fps']:.3f}",
                      f"{res['speed_px_s'][i]:.3f}",
                      "" if res["speed_m_s"] is None else f"{res['speed_m_s'][i]:.3f}",
                      f"{res['angle_deg'][i]:.2f}"])
    return F0.guardar_csv(
        filas, ["frame", "t_s", "vel_px_s", "vel_m_s", "angulo_imagen_deg"],
        ruta_csv)


def video_flujo(ruta_video, masks, ruta_salida, fps, escala=0.5, gsd_m_px=None,
                max_gap=3, paso_grilla=22, mag_min=0.6, escala_flecha=4.0,
                min_px=200, alfa=0.35, n_paneles=0, ruta_png=None,
                pivot_frame=None, cols=3):
    """Mascara (rojo) + campo de flujo (verde) + direccion dominante (amarillo)."""
    fkeys = sorted(masks)
    H, W = masks[fkeys[0]].shape
    pares = [k for k in range(1, len(fkeys)) if fkeys[k] - fkeys[k - 1] <= max_gap]
    paso = float(np.median([fkeys[k] - fkeys[k - 1] for k in pares])) if pares else 1.0
    fps_video = fps / paso
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps_video, W, H)
    idx_fig = set()
    if n_paneles and pares:
        idx_fig = {pares[i] for i in np.unique(
            np.linspace(0, len(pares) - 1, n_paneles).round().astype(int))}
    paneles = []
    n = 0
    with F0.LectorVideo(ruta_video, escala) as lec:
        for k in range(1, len(fkeys)):
            fi, fi_prev = fkeys[k], fkeys[k - 1]
            salto = fi - fi_prev
            if salto > max_gap:
                continue
            small, small_prev = lec[fi], lec[fi_prev]
            if small is None or small_prev is None:
                continue
            small, small_prev = small[:Hv, :Wv], small_prev[:Hv, :Wv]
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gp = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
            m = F0.normalizar_mascara(masks[fi], g.shape)
            mp = F0.normalizar_mascara(masks[fi_prev], g.shape)
            comun = cv2.bitwise_and(m, mp)
            flow = cv2.calcOpticalFlowFarneback(gp, g, None, pyr_scale=0.5,
                                                levels=3, winsize=21, iterations=3,
                                                poly_n=7, poly_sigma=1.5, flags=0)
            capa = small.copy(); capa[m > 0] = (0, 0, 255)
            vis = cv2.addWeighted(capa, alfa, small, 1 - alfa, 0)
            for gy in range(0, g.shape[0], paso_grilla):
                for gx in range(0, g.shape[1], paso_grilla):
                    if comun[gy, gx] == 0:
                        continue
                    fxp, fyp = flow[gy, gx]
                    if np.hypot(fxp, fyp) < mag_min:
                        continue
                    cv2.arrowedLine(vis, (gx, gy),
                                    (int(gx + fxp * escala_flecha),
                                     int(gy + fyp * escala_flecha)),
                                    (0, 255, 0), 1, tipLength=0.3)
            sel = comun > 0
            if int(sel.sum()) > min_px:
                fx, fy = flow[..., 0][sel], flow[..., 1][sel]
                eje = direccion_dominante(fx, fy)
                p = float(np.median(fx * eje[0] + fy * eje[1])) / salto
                vel = abs(p) / escala * fps
                ang = (-np.degrees(np.arctan2(eje[1], eje[0]))) % 360
                M = cv2.moments(sel.astype(np.uint8), binaryImage=True)
                cxm, cym = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.arrowedLine(vis, (cxm, cym),
                                (int(cxm + eje[0] * 70), int(cym + eje[1] * 70)),
                                (0, 220, 255), 3, tipLength=0.3)
                txt = f"{vel*gsd_m_px:.1f} m/s" if gsd_m_px else f"{vel:.0f} px/s"
                etiqueta = f"dir {ang:.0f} deg   {txt}"
                texto, color = f"f{fi}  dir:{ang:.0f}  {txt}", (0, 220, 255)
            else:
                etiqueta = "pocos px"
                texto, color = f"f{fi}  (pocos px)", (200, 200, 200)
            if k in idx_fig:
                paneles.append((fi, etiqueta,
                                cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)))
            cv2.putText(vis, texto, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            vw.write(vis); n += 1
    vw.release()
    print(f"[video] {n} frames a {fps_video:.2f} fps (paso {paso:.0f}) -> {F0.ruta_corta(ruta_salida)}")
    fig = None
    if paneles:
        fig = paneles_flujo(paneles, fps, pivot_frame=pivot_frame,
                            ruta_png=ruta_png, cols=cols)
    return ruta_salida, fig


def paneles_flujo(paneles, fps, pivot_frame=None, ruta_png=None, cols=3,
                  margen=0.06):
    """Figura de `n` frames del video de flujo, recortados al mismo encuadre."""
    import matplotlib.pyplot as plt

    alto, ancho = paneles[0][3].shape[:2]
    x0, y0, x1, y1 = ancho, alto, 0, 0
    for _, _, base, vis in paneles:
        d = np.abs(vis.astype(np.int16) - base.astype(np.int16)).max(2) > 12
        ys, xs = np.nonzero(d)
        if xs.size:
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, ancho - 1, alto - 1
    mx, my = int(margen * ancho), int(margen * alto)
    x0, x1 = max(0, x0 - mx), min(ancho - 1, x1 + mx)
    y0, y1 = max(0, y0 - my), min(alto - 1, y1 + my)
    lado = (x1 - x0) / max(1, y1 - y0)

    filas = int(np.ceil(len(paneles) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(4.4 * cols, 4.4 * filas / lado),
                            constrained_layout=True)
    ejes = np.atleast_1d(axs).ravel()
    for ax, (fi, etiqueta, _, vis) in zip(ejes, paneles):
        ax.imshow(vis[y0:y1 + 1, x0:x1 + 1])
        tt = f"t = {(fi - pivot_frame) / fps:.1f} s" if pivot_frame is not None \
            else f"frame {fi}"
        ax.set_title(f"{tt}   |   {etiqueta}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in ejes[len(paneles):]:
        ax.axis("off")
    if ruta_png:
        fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
        print(f"[fig] {len(paneles)} paneles -> {F0.ruta_corta(ruta_png)}")
    return fig


CamaraGeo = F0.CamaraGeo


def altura_desde_glb(ruta_glb, z_piso, ruta_serie_columna=None, verbose=True):
    """Centroide 3D del hull por keyframe, desde `gas_secuencia.glb` (Fase 2)."""
    import re

    import trimesh

    esc = trimesh.load(ruta_glb)
    reg = {}
    for nombre, m in esc.geometry.items():
        f = int(re.search(r"(\d+)", nombre).group(1))
        V = np.asarray(m.vertices, float)
        try:
            c = m.center_mass if m.is_volume else V.mean(0)
        except Exception:
            c = V.mean(0)
        reg[f] = (float(c[0]), float(-c[2]), float(c[1]), float(V[:, 1].max()))

    fr = np.array(sorted(reg))
    E = np.array([reg[f][0] for f in fr])
    N = np.array([reg[f][1] for f in fr])
    Uu = np.array([reg[f][2] for f in fr])
    tope = np.array([reg[f][3] for f in fr])

    conf = np.ones(len(fr), bool)
    if ruta_serie_columna and os.path.exists(ruta_serie_columna):
        import csv as _csv
        with open(ruta_serie_columna, "r", encoding="utf-8") as fh:
            mapa = {int(r["frame"]): r.get("modo", "") != "sin_sombra"
                    for r in _csv.DictReader(fh)}
        if mapa:
            c_ = np.array([mapa.get(int(f), False) for f in fr])
            if c_.any():
                conf = c_
            elif verbose:
                print("  [aviso] ningun keyframe con tallado solar; uso todos")

    if verbose:
        r = np.nanmedian(Uu / np.where(tope > 0, tope, np.nan))
        print(f"  {len(fr)} keyframes [f{fr[0]}, f{fr[-1]}] | confiables {int(conf.sum())}")
        print(f"  centroide U: {Uu.min():.1f} - {Uu.max():.1f} m sobre el piso "
              f"({r:.2f} del tope)")
    return dict(frames=fr, E=E, N=N, U=Uu, tope=tope, conf=conf, z_adv=z_piso + Uu)


def altura_interpolada(frames, alt, solo_confiables=True, suavizar=7):
    """Interpola z_adv a `frames`; fuera del tramo confiable mantiene el ultimo valor.
    """
    fr, z = alt["frames"], alt["z_adv"]
    if solo_confiables and alt["conf"].any():
        fr, z = fr[alt["conf"]], z[alt["conf"]]
    if suavizar and len(z) >= suavizar:
        z = uniform_filter1d(z, size=suavizar, mode="nearest")
    return np.interp(np.asarray(frames, float), fr, z), (int(fr.min()), int(fr.max()))


def resolver_altura(cfg, frames, cam, ruta_glb="auto", ruta_serie="auto",
                    altura_fija_m=25.0, verbose=True):
    """Cota del plano de adveccion por frame."""
    z0 = float(cam.z_piso)
    banderas = []

    if ruta_glb == "auto":
        ruta_glb = cfg.art_gas_glb
    if ruta_serie == "auto":
        ruta_serie = cfg.art_serie_columna

    if ruta_glb and os.path.exists(ruta_glb):
        try:
            alt = altura_desde_glb(ruta_glb, z_piso=z0,
                                   ruta_serie_columna=ruta_serie, verbose=verbose)
            z, (f0, f1) = altura_interpolada(frames, alt)
            info = dict(fuente="hull 3D de la Fase 2", alt=alt,
                        frames_confiables=(f0, f1), banderas=banderas)
            if verbose:
                print(f"  altura: hull 3D, tramo confiable f{f0}-{f1}")
            return z, info
        except Exception as e:
            banderas.append(f"hull_ilegible:{type(e).__name__}")
            if verbose:
                print(f"  [aviso] no pude usar el hull ({e}); caigo a altura fija")

    banderas.append(f"[i] sin hull 3D: plano fijo a {altura_fija_m:.0f} m")
    z = np.full(len(frames), z0 + float(altura_fija_m))
    if verbose:
        print(f"  altura: plano fijo a {altura_fija_m:.0f} m sobre el piso "
              f"(no hay hull de la Fase 2)")
        print(f"    la direccion NO se ve afectada (homotecia); la velocidad "
              f"sale con banda -> C9")
    return z, dict(fuente=f"altura fija {altura_fija_m:.0f} m", alt=None,
                   frames_confiables=None, banderas=banderas)


def diagnostico_bimodal(res, t, t_ini=None, t_fin=None, kappa_kde=180.0,
                        n_grilla=720, dip_min=0.30, alto_min=0.25,
                        sep_min_deg=15.0, verbose=True):
    """Alerta de bimodalidad: dos montanas marcadas en la distribucion angular."""
    t = np.asarray(t, float)
    m = (np.ones_like(t, bool) if t_ini is None else
         (t >= t_ini) & (t <= (np.inf if t_fin is None else t_fin)))
    ang = np.asarray(res["vmf_muestra"], float)[m].ravel()
    ang = ang[np.isfinite(ang)]
    vacio = dict(bimodal=False, sep_deg=np.nan, alto=np.nan, dip=np.nan,
                 modos_deg=[], n=int(ang.size))
    if ang.size < 100:
        return vacio

    aa = np.linspace(-np.pi, np.pi, n_grilla, endpoint=False)
    kde = np.array([np.exp(kappa_kde * np.cos(x - ang)).sum() for x in aa])

    izq, der = np.roll(kde, 1), np.roll(kde, -1)
    pic = np.flatnonzero((kde > izq) & (kde >= der))
    if pic.size < 2:
        if verbose:
            print("  bimodalidad: un solo modo")
        return vacio
    i1, i2 = pic[np.argsort(kde[pic])[::-1][:2]]

    n1 = (i2 - i1) % n_grilla
    arco = ((np.arange(i1, i1 + n1 + 1) if n1 <= n_grilla - n1
             else np.arange(i2, i2 + (n_grilla - n1) + 1)) % n_grilla)
    sep = 360.0 * min(n1, n_grilla - n1) / n_grilla
    alto = float(kde[i2] / kde[i1])
    dip = float(1.0 - kde[arco].min() / kde[i2])
    bim = bool(dip > dip_min and alto > alto_min and sep > sep_min_deg)

    out = dict(bimodal=bim, sep_deg=float(sep), alto=alto, dip=dip,
               modos_deg=[float(np.degrees(aa[i1])), float(np.degrees(aa[i2]))],
               n=int(ang.size))
    if verbose:
        print(f"  modos en {out['modos_deg'][0]:.1f} y {out['modos_deg'][1]:.1f} deg"
              f"  |  sep {sep:.1f} deg, alto {alto:.2f}, dip {dip:.2f}")
        print(f"  -> {'BIMODAL: la media cae en el valle' if bim else 'unimodal'}"
              f"  (umbrales dip>{dip_min:.2f}, alto>{alto_min:.2f}, "
              f"sep>{sep_min_deg:.0f} deg)")
    return out


def banderas_calidad(cfg, cam, geo, t, t_ini, t_fin, t_fin_vel, info_altura,
                     escala=None, deriva=None, diag=None, res=None, fps=None):
    """Resumen de en que confiar y en que no. Va al CSV y al resumen impreso."""
    b = list(info_altura.get("banderas", []))
    n = int(((t >= t_ini) & (t <= t_fin)).sum())
    if n < 30:
        b.append(f"pocos_pares_direccion:{n}")
    nv = int(((t >= t_ini) & (t <= t_fin_vel)).sum())
    if nv < 15:
        b.append(f"pocos_pares_velocidad:{nv}")
    if t_fin - t_ini < 2.0:
        b.append("ventana_corta")
    if diag:
        if diag.get("degenerada"):
            b.append("ventana_degenerada")
        if "textura" not in diag and "borde" not in diag:
            b.append("ventana_sin_criterio_de_fin")
        elif diag.get("textura", {}).get("t") is None \
                and diag.get("borde", {}).get("t") is None:
            b.append("fin_por_fin_de_clip")
        ta = diag.get("t_altura_confiable")
        if ta is not None and ta < t_fin - 0.5:
            b.append(f"altura_sostenida_desde:{ta:.1f}s")
    if res is not None and fps:
        m = (t >= t_ini) & (t <= t_fin)
        fr = np.asarray(res["frames"], float)
        salto = float(np.median(np.diff(fr))) if len(fr) > 1 else 1.0
        if m.any():
            px_par = (float(np.median(np.asarray(res["speed_px_s"], float)[m]))
                      * float(cam.escala) / float(fps) * max(salto, 1.0))
            if px_par < 1.0:
                b.append(f"desplazamiento_subpixel:{px_par:.2f}px/par")
    if res is not None and "vmf_muestra" in res:
        bm = diagnostico_bimodal(res, t, t_ini, t_fin, verbose=False)
        if bm["bimodal"]:
            b.append(f"bimodal:sep={bm['sep_deg']:.0f}deg,dip={bm['dip']:.2f}")
    disp = dispersion_circular_deg(geo["azimut_hacia"][(t >= t_ini) & (t <= t_fin)],
                                   geo["vel_ms"][(t >= t_ini) & (t <= t_fin)])
    if disp > 25:
        b.append(f"direccion_dispersa:{disp:.0f}deg")
    if escala and abs(escala.get("error_pct", 0)) > 10:
        b.append(f"escala_discrepa:{escala['error_pct']:+.0f}%")
    if deriva:
        if deriva.get("horizontal_m", 0) > 2.0:
            b.append(f"dron_se_movio:{deriva['horizontal_m']:.1f}m")
        if deriva.get("yaw_deg", 0) > 3.0:
            b.append(f"yaw_vario:{deriva['yaw_deg']:.1f}deg")
    if cam.sol and cam.sol["elevacion_deg"] > 60:
        b.append("sol_alto_sombra_corta")
    graves = [x for x in b if not str(x).startswith("[i]")]
    calidad = "ok" if not graves else ("revisar" if len(graves) <= 2 else "dudoso")
    return dict(calidad=calidad, banderas=b, fuente_altura=info_altura["fuente"])


def georreferenciar(res, cam, u_px, v_px, z_adv, eps=5.0):
    """Serie del plano imagen -> azimut geografico + m/s."""
    ang = np.radians(np.asarray(res["angle_deg"], float))
    u = np.asarray(u_px, float); v = np.asarray(v_px, float)
    z = np.asarray(z_adv, float)

    A = cam.a_plano(u, v, z)
    B = cam.a_plano(u + eps * np.cos(ang), v + eps * np.sin(ang), z)
    d = B - A

    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
    mpp = np.linalg.norm(d[:, :2], axis=1) / eps * cam.escala
    return dict(frames=np.asarray(res["frames"]),
                azimut_hacia=az, azimut_desde=(az + 180.0) % 360.0,
                vel_ms=np.asarray(res["speed_px_s"], float) * mpp,
                m_por_px=mpp, u_px=u, v_px=v, z_adv=z,
                E=A[:, 0], N=A[:, 1],
                angulo_imagen=np.asarray(res["angle_deg"], float))


def sensibilidad_altura(res, cam, u_px, v_px, z_base, t, deltas=(-10, -5, 0, 5, 10),
                        t_ini=8.0, t_fin=None):
    """Como cambia la estimacion si el plano de adveccion sube o baja."""
    m = (t >= t_ini) & (t <= (t.max() if t_fin is None else t_fin))
    print(f"  {'dz (m)':>7} {'HACIA':>8} {'DESDE':>8} {'vel m/s':>9}")
    filas = []
    for dz in deltas:
        g = georreferenciar(res, cam, u_px, v_px, np.asarray(z_base) + dz)
        h = media_circular_pond(g["azimut_hacia"][m], g["vel_ms"][m])
        vm = float(g["vel_ms"][m].mean())
        filas.append((dz, h, (h + 180) % 360, vm))
        print(f"  {dz:>7.0f} {h:>8.1f} {(h+180)%360:>8.1f} {vm:>9.2f}")
    return filas


def _primer_sostenido(t, serie, umbral, sostener, desde=None, sentido="menor"):
    """Primer instante en que `serie` cruza `umbral` y se queda cruzada."""
    t = np.asarray(t, float); s = np.asarray(serie, float)
    cruza = (s < umbral) if sentido == "menor" else (s > umbral)
    for i in range(len(t)):
        if desde is not None and t[i] < desde:
            continue
        if not cruza[i] or not np.isfinite(s[i]):
            continue
        w = (t >= t[i]) & (t <= t[i] + sostener)
        if w.sum() > 1 and cruza[w].all():
            return float(t[i])
    return None


def detectar_ventana(res, t, hull=None, masks=None, forma=None,
                     umbral_ascenso=0.30, umbral_deriva=5.0, sostener=2.0,
                     frac_textura=0.30, margen_borde=0.10, frac_recorte=0.02,
                     sostener_borde=0.5, suavizar_metricas=9,
                     frac_area=None, verbose=True):
    """Decide DONDE empieza y termina el viento establecido, con criterios medibles."""
    t = np.asarray(t, float)
    diag = {}

    tA = None
    if hull is not None:
        r = np.abs(hull["vU"]) / np.maximum(hull["vel_ms"], 1e-6)
        th = hull["t"]
        for i in range(len(th)):
            if r[i] >= umbral_ascenso:
                continue
            w = (th >= th[i]) & (th <= th[i] + sostener)
            if w.sum() > 1 and (r[w] < umbral_ascenso).all():
                tA = float(th[i]); break
        diag["ascenso"] = dict(t=tA, umbral=umbral_ascenso, serie_t=th, serie=r)

    ang = np.degrees(np.unwrap(np.radians(np.asarray(res["angle_deg"], float))))
    deriva = np.abs(np.gradient(ang) / np.gradient(t))
    tB = _primer_sostenido(t, deriva, umbral_deriva, sostener)
    diag["deriva"] = dict(t=tB, umbral=umbral_deriva, serie_t=t, serie=deriva)

    t_ini = max([x for x in (tA, tB) if x is not None], default=float(t.min()))

    tC = None
    if "textura" in res:
        tex = np.asarray(res["textura"], float)
        if suavizar_metricas and len(tex) >= suavizar_metricas:
            tex = uniform_filter1d(tex, size=suavizar_metricas, mode="nearest")
        tC = _primer_sostenido(t, tex, frac_textura, sostener, desde=t_ini)
        diag["textura"] = dict(t=tC, umbral=frac_textura, serie_t=t, serie=tex,
                               cnr=np.asarray(res.get("cnr", []), float))

    tD = None
    if "dist_borde" in res:
        db = np.asarray(res["dist_borde"], float)
        rec = np.asarray(res.get("recorte", np.zeros(len(db))), float)
        if suavizar_metricas and len(db) >= suavizar_metricas:
            db = uniform_filter1d(db, size=suavizar_metricas, mode="nearest")
        lado = min(res["forma"]) if res.get("forma") else 2.0 * db.max()
        umb = margen_borde * lado if margen_borde <= 1.0 else margen_borde
        t1 = _primer_sostenido(t, db, umb, sostener_borde, desde=t_ini)
        t2 = _primer_sostenido(t, rec, frac_recorte, sostener_borde, desde=t_ini,
                               sentido="mayor")
        tD = min([x for x in (t1, t2) if x is not None], default=None)
        diag["borde"] = dict(t=tD, t_dist=t1, t_recorte=t2, umbral=umb,
                             umbral_recorte=frac_recorte, serie_t=t, serie=db,
                             recorte=rec)

    if masks is not None and frac_area is not None:
        area = np.array([float(np.sum(
            (F0.normalizar_mascara(masks[int(f)], forma) if forma is not None
             else np.asarray(masks[int(f)])) > 0)) for f in res["frames"]])
        pico_a = int(np.argmax(area))
        bajo = np.where((np.arange(len(area)) > pico_a)
                        & (area < frac_area * area[pico_a]))[0]
        diag["area"] = dict(serie_t=t, serie=area, pico=pico_a, frac=frac_area,
                            t=float(t[bajo[0]]) if len(bajo) else None)
    elif "area" in res:
        diag["area"] = dict(serie_t=t, serie=np.asarray(res["area"], float),
                            pico=int(np.argmax(res["area"])), frac=None, t=None)

    t_fin = min([x for x in (tC, tD) if x is not None], default=float(t.max()))
    if t_fin <= t_ini + 1e-9:
        t_fin = float(t.max())
        diag["degenerada"] = (f"el criterio de fin cae en t_ini ({t_ini:.2f} s): "
                              f"no hay ventana util en este video")

    t_fin_vel = t_fin
    t_alt = float(hull["t"].max()) if hull is not None else None
    diag["t_altura_confiable"] = t_alt

    if verbose:
        def _f(x):
            return f"t = {x:.2f} s" if x is not None else "no se alcanza"
        print(f"  A) columna dejo de subir   (|w|/|u| < {umbral_ascenso:.2f})  : "
              f"{_f(tA) if hull is not None else 'sin hull 3D'}")
        print(f"  B) direccion dejo de girar (< {umbral_deriva:.0f} deg/s)      : {_f(tB)}")
        print(f"  -> t_ini = {t_ini:.2f} s  (el mas tardio de los dos)")
        falta = "res sin metricas -> vuelve a correr C2 (estimar_viento)"
        print(f"  C) textura bajo {frac_textura:.2f} (paridad 0.50)   : "
              f"{_f(tC) if 'textura' in diag else falta}")
        print(f"  D) frente a < {diag.get('borde', {}).get('umbral', 0):.0f} px del borde   : "
              f"{_f(tD) if 'borde' in diag else falta}")
        print(f"  -> t_fin = {t_fin:.2f} s  (el mas temprano de los dos)")
        if diag.get("degenerada"):
            print(f"  [OJO] {diag['degenerada']}")
        if diag.get("area", {}).get("t") is not None:
            print(f"     [diag] el area de mascara habria cortado en "
                  f"{diag['area']['t']:.2f} s")
        elif "area" in diag:
            print("     [diag] el area de mascara nunca cae bajo el umbral "
                  "(SAM2 sigue segmentando la nube diluida)")
        if t_alt is not None and t_alt < t_fin - 0.5:
            print(f"  [aviso] el hull 3D termina en {t_alt:.2f} s: mas alla de ahi "
                  f"la altura se sostiene, no se mide.")
    return t_ini, t_fin, t_fin_vel, diag


def resumen_ventana(geo, t, t_ini, t_fin=None, etiqueta="ventana"):
    """Resumen del viento en una ventana, ponderado por velocidad."""
    tf = float(t.max()) if t_fin is None else float(t_fin)
    s = (t >= t_ini) & (t <= tf)
    if not s.any():
        print(f"  {etiqueta}: sin datos"); return None
    v = geo["vel_ms"][s]
    hacia = media_circular_pond(geo["azimut_hacia"][s], v)
    out = dict(n=int(s.sum()), t_ini=float(t_ini), t_fin=tf,
               hacia=hacia, desde=(hacia + 180) % 360,
               dispersion=dispersion_circular_deg(geo["azimut_hacia"][s], v),
               vel_media=float(v.mean()), vel_p50=float(np.median(v)),
               vel_min=float(v.min()), vel_max=float(v.max()), mascara=s)
    print(f"  {etiqueta}  t = [{t_ini:.1f}, {tf:.1f}] s   n = {out['n']}")
    print(f"    HACIA {out['hacia']:6.1f} deg  |  DESDE {out['desde']:6.1f} deg"
          f"   (dispersion {out['dispersion']:.1f} deg)")
    print(f"    velocidad {out['vel_media']:.2f} m/s  "
          f"(mediana {out['vel_p50']:.2f}, rango {out['vel_min']:.2f}-{out['vel_max']:.2f})")
    return out


cargar_dem = F0.cargar_dem


def exportar_series_simples(res, t, fps, ruta_video_csv, geo=None,
                            ruta_mundo_csv=None, t_ini=None, t_fin=None,
                            t_fin_vel=None, solo_ventana=False, verbose=True):
    """Dos CSV planos, al estilo del que entrega la DMC: tres columnas y listo."""
    t = np.asarray(t, float)
    frames = np.asarray(res["frames"])
    ti = float(t.min()) if t_ini is None else float(t_ini)
    tf = float(t.max()) if t_fin is None else float(t_fin)
    tv = tf if t_fin_vel is None else float(t_fin_vel)

    vel_px_frame = np.asarray(res["speed_px_s"], float) / float(fps)
    dir_video = np.asarray(res["angle_deg"], float) % 360.0

    m_dir = (t >= ti) & (t <= tf)
    m_ms = (t >= ti) & (t <= tv)
    if not m_dir.any():
        raise ValueError("La ventana de viento no contiene ninguna muestra.")
    sel = m_dir if solo_ventana else np.ones(len(t), bool)

    def _fila_promedio(v, ang, mv, ma, pesos):
        if solo_ventana:
            etq = "PROMEDIO"
        elif np.array_equal(mv, ma):
            etq = f"PROMEDIO f{frames[ma][0]}-f{frames[ma][-1]}"
        else:
            etq = (f"PROMEDIO vel f{frames[mv][0]}-f{frames[mv][-1]} "
                   f"dir f{frames[ma][0]}-f{frames[ma][-1]}")
        return [etq, f"{float(np.mean(v[mv])):.4f}",
                f"{media_circular_pond(ang[ma], pesos[ma]):.1f}"]

    salidas = {}
    filas = [[int(f), f"{vel_px_frame[i]:.4f}", f"{dir_video[i]:.1f}"]
             for i, f in enumerate(frames) if sel[i]]
    filas.append(_fila_promedio(vel_px_frame, dir_video, m_dir, m_dir, vel_px_frame))
    salidas["video"] = F0.guardar_csv(
        filas, ["frame", "vel_px_frame", "dir_video_deg"], ruta_video_csv)

    if geo is not None and ruta_mundo_csv:
        vel_ms = np.asarray(geo["vel_ms"], float)
        dir_desde = np.asarray(geo["azimut_desde"], float) % 360.0
        filas = [[int(f), f"{vel_ms[i]:.4f}", f"{dir_desde[i]:.1f}"]
                 for i, f in enumerate(frames) if sel[i]]
        filas.append(_fila_promedio(vel_ms, dir_desde, m_ms, m_dir, vel_ms))
        salidas["mundo"] = F0.guardar_csv(
            filas, ["frame", "vel_ms", "dir_desde_deg"], ruta_mundo_csv)

    if verbose:
        vpf = float(np.mean(vel_px_frame[m_dir]))
        dv = media_circular_pond(dir_video[m_dir], vel_px_frame[m_dir])
        print(f"  ventana de viento establecido: t = [{ti:.1f}, {tf:.1f}] s")
        print(f"    video : {vpf:.3f} px/frame   dir {dv:.1f} deg "
              f"(0 = derecha, horario en pantalla)")
        if geo is not None:
            vm = float(np.mean(np.asarray(geo["vel_ms"])[m_ms]))
            dd = media_circular_pond(np.asarray(geo["azimut_desde"])[m_dir],
                                     np.asarray(geo["vel_ms"])[m_dir])
            print(f"    mundo : {vm:.3f} m/s        DESDE {dd:.1f} deg "
                  f"(0 = Norte, horario)")
            if tv < tf:
                print(f"            los m/s se promedian solo hasta t = {tv:.1f} s "
                      f"(t_fin_vel forzado a mano);")
                print(f"            px/frame no depende de la altura y usa la "
                      f"ventana completa")
        if not solo_ventana:
            print("    [ojo] la fila PROMEDIO no es el promedio de las filas de "
                  "arriba: se calcula solo en la ventana")
    return salidas


def _a_local(ts_utc, tz_local):
    import datetime
    from zoneinfo import ZoneInfo
    return (ts_utc.replace(tzinfo=datetime.timezone.utc)
            .astimezone(ZoneInfo(tz_local)).replace(tzinfo=None))


def _utm_estacion(coords_latlon=None, coords_utm=None, crs="EPSG:32719"):
    """Ubicacion de la estacion en UTM. Acepta (lat, lon, alt) o (E, N, alt)."""
    if coords_utm is not None:
        return tuple(float(x) if x is not None else None for x in coords_utm)
    if coords_latlon is None:
        return None
    from pyproj import Transformer
    lat, lon = float(coords_latlon[0]), float(coords_latlon[1])
    alt = coords_latlon[2] if len(coords_latlon) > 2 else None
    E, N = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return (E, N, float(alt) if alt is not None else None)


def _leer_tabla_dmc(ruta):
    """Una tabla mensual de la DMC -> (ts_utc, dd_deg, ff_ms, unidad_original)."""
    import pandas as pd
    tablas = pd.read_html(ruta)
    if not tablas:
        raise ValueError(f"{os.path.basename(ruta)} no contiene ninguna tabla.")
    d = tablas[0]
    d.columns = [str(c).strip().lower() for c in d.columns]
    col = {}
    for c in d.columns:
        if c.startswith("fecha"): col["fecha"] = c
        elif c.startswith("hora"): col["hora"] = c
        elif c.startswith("dd"): col["dd"] = c
        elif c.startswith("ff"): col["ff"] = c
    faltan = {"fecha", "hora", "dd", "ff"} - set(col)
    if faltan:
        raise ValueError(f"A {os.path.basename(ruta)} le faltan columnas "
                         f"{faltan}. Encontre: {list(d.columns)}")
    ts_utc = pd.to_datetime(d[col["fecha"]].astype(str) + " " +
                            d[col["hora"]].astype(str),
                            format="%d-%m-%Y %H:%M", errors="coerce")
    en_nudos = "kt" in col["ff"] or "nudo" in col["ff"]
    ff = pd.to_numeric(d[col["ff"]], errors="coerce") * (0.514444 if en_nudos else 1.0)
    dd = pd.to_numeric(d[col["dd"]], errors="coerce")
    ok = ts_utc.notna()
    return (list(ts_utc[ok]), np.asarray(dd[ok], float), np.asarray(ff[ok], float),
            "nudos" if en_nudos else "m/s")


def cargar_estacion_dmc(rutas, tz_local="America/Santiago", nombre="DMC",
                        coords_latlon=None, coords_utm=None, crs="EPSG:32719",
                        verbose=True):
    """Estacion de la DMC (HTML con extension .xls). Hora LOCAL y m/s."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("El archivo de la DMC es HTML; hace falta pandas "
                          "(y lxml o html5lib) para leerlo.")
    if isinstance(rutas, (str, bytes, os.PathLike)):
        rutas = [rutas]
    rutas = list(rutas)
    if not rutas:
        raise ValueError(f"{nombre}: no me pasaste ningun archivo.")

    por_ts, unidad, leidos, malos, discrepan = {}, None, [], [], []
    for r in sorted(rutas):
        try:
            ts_utc, dd, ff, u = _leer_tabla_dmc(r)
        except Exception as ex:
            malos.append((os.path.basename(r), f"{type(ex).__name__}: {ex}"))
            continue
        unidad = unidad or u
        dentro = sorted({f"{t.year:04d}-{t.month:02d}" for t in ts_utc})
        m_nom = re.search(r"(?<!\d)(20\d{2})[-_]?(0[1-9]|1[0-2])(?!\d)",
                          os.path.basename(r))
        if m_nom:
            pide = f"{m_nom.group(1)}-{m_nom.group(2)}"
            if pide not in dentro:
                discrepan.append((os.path.basename(r), pide, dentro))
        for t, a, v in zip(ts_utc, dd, ff):
            por_ts[t.to_pydatetime()] = (a, v)
        leidos.append((os.path.basename(r), len(ts_utc)))
    if not por_ts:
        det = "; ".join(f"{f}: {m}" for f, m in malos) or "sin filas utiles"
        raise ValueError(f"{nombre}: no pude leer ninguna tabla ({det}).")

    orden = sorted(por_ts)
    est = dict(nombre=nombre, fuente="DMC", tz=tz_local,
               ts=[_a_local(t, tz_local) for t in orden],
               dir=np.array([por_ts[t][0] for t in orden], float),
               vel=np.array([por_ts[t][1] for t in orden], float),
               unidad_original=unidad, hora_original="UTC",
               archivos=[os.path.basename(r) for r in sorted(rutas)],
               utm=_utm_estacion(coords_latlon, coords_utm, crs))
    if verbose:
        if len(leidos) > 1:
            print(f"  {nombre}: {len(leidos)} archivos mensuales -> "
                  f"{len(orden)} instantes unicos")
        for f, m in malos:
            print(f"    [aviso] {f} ilegible ({m}); se ignora")
        for f, pide, dentro in discrepan:
            print(f"    [AVISO] {f} se llama como si fuera {pide}, pero por "
                  f"dentro trae {', '.join(dentro)}.")
            print(f"            En el portal de la DMC el selector de ANO se "
                  f"queda donde estaba: revisa que bajaste {pide} y no el "
                  f"mismo mes de otro ano.")
        _resumen_estacion(est)
        _meses_dmc(est, verbose=True)
    return est


def _meses_dmc(est, verbose=True):
    """Los meses que trae la serie y los huecos entre ellos."""
    meses = sorted({f"{t.year:04d}-{t.month:02d}" for t in est["ts"]})
    if not meses:
        return meses
    a0, m0 = int(meses[0][:4]), int(meses[0][5:])
    a1, m1 = int(meses[-1][:4]), int(meses[-1][5:])
    todos = []
    while (a0, m0) <= (a1, m1):
        todos.append(f"{a0:04d}-{m0:02d}")
        m0 += 1
        if m0 == 13:
            a0, m0 = a0 + 1, 1
    huecos = [m for m in todos if m not in meses]
    if verbose:
        print(f"    meses: {', '.join(meses)}")
        if huecos:
            print(f"    SIN DATOS en: {', '.join(huecos)}")
    return meses


def cargar_estacion_sinca(ruta_dir, ruta_vel, nombre="SINCA",
                          coords_latlon=None, coords_utm=None,
                          crs="EPSG:32719", verbose=True):
    """Los dos CSV de SINCA (direccion y velocidad). Hora LOCAL, m/s."""
    import datetime

    def _leer(ruta, que):
        r, saltadas, muestra = {}, 0, None
        with open(ruta, "r", encoding="latin1") as fh:
            for i, linea in enumerate(fh):
                if i == 0:
                    continue
                c = linea.rstrip("\n").split(";")
                if len(c) < 3 or not c[0].strip():
                    if linea.strip():
                        saltadas += 1
                        muestra = muestra or linea.strip()[:40]
                    continue
                f, h = c[0].strip(), c[1].strip()
                if not (f.isdigit() and len(f) == 6 and h.isdigit()):
                    raise ValueError(
                        f"{os.path.basename(ruta)} no parece una serie de {que}: "
                        f"la fila {i+1} es '{linea.strip()[:40]}'.\n"
                        f"    En SINCA es facil bajar la tabla de la ROSA DE "
                        f"VIENTOS en vez de la serie horaria; tienen la misma "
                        f"cabecera. Vuelve a exportar como serie de tiempo.")
                v = c[2].strip().replace(",", ".")
                ts = datetime.datetime(2000 + int(f[:2]), int(f[2:4]), int(f[4:6]),
                                       int(h[:2]), int(h[2:]))
                r[ts] = float(v) if v else np.nan
        if not r:
            raise ValueError(
                f"{os.path.basename(ruta)} no contiene ninguna serie de {que} "
                f"({saltadas} filas sin el formato FECHA;HORA;valor"
                + (f", p.ej. '{muestra}'" if muestra else "") + ").\n"
                f"    En SINCA es facil bajar la tabla de la ROSA DE VIENTOS en "
                f"vez de la serie horaria: tienen la misma cabecera pero la rosa "
                f"trae sectores y porcentajes.\n"
                f"    Vuelve a exportar eligiendo la serie de tiempo.")
        return r

    D, V = _leer(ruta_dir, "direccion"), _leer(ruta_vel, "velocidad")
    ts = sorted(set(D) & set(V))
    est = dict(nombre=nombre, fuente="SINCA", tz="America/Santiago", ts=ts,
               dir=np.array([D[t] for t in ts], float),
               vel=np.array([V[t] for t in ts], float),
               unidad_original="m/s", hora_original="local",
               utm=_utm_estacion(coords_latlon, coords_utm, crs))
    if verbose:
        _resumen_estacion(est)
    return est


def _resumen_estacion(est):
    n = len(est["ts"])
    hd = int(np.isnan(est["dir"]).sum()); hv = int(np.isnan(est["vel"]).sum())
    print(f"  {est['nombre']} ({est['fuente']}): {n} registros | "
          f"{est['ts'][0]:%Y-%m-%d} .. {est['ts'][-1]:%Y-%m-%d} (hora local)")
    print(f"    huecos: direccion {hd} ({100*hd/max(n,1):.1f} %), "
          f"velocidad {hv} ({100*hv/max(n,1):.1f} %)")
    v = est["vel"][~np.isnan(est["vel"])]
    if len(v) and est["unidad_original"] == "nudos":
        paso = 0.514444
        print(f"    velocidad cuantizada a nudos enteros: pasos de {paso:.2f} m/s")


def viento_en(est, ts, max_hueco_h=2.0):
    """Viento interpolado al instante `ts` (datetime u hora ISO local)."""
    import datetime
    if isinstance(ts, str):
        ts = datetime.datetime.fromisoformat(ts)
    T = est["ts"]
    i = int(np.searchsorted(np.array([t.timestamp() for t in T]), ts.timestamp()))
    if i == 0 or i >= len(T):
        raise ValueError(f"{ts} queda fuera del rango de {est['nombre']} "
                         f"({T[0]} .. {T[-1]}).")
    a, b = T[i - 1], T[i]
    dt = (b - a).total_seconds() / 3600.0
    if dt > max_hueco_h:
        raise ValueError(f"Hueco de {dt:.1f} h alrededor de {ts} en "
                         f"{est['nombre']}: no interpolo.")
    w = (ts - a).total_seconds() / (b - a).total_seconds()
    da, db = est["dir"][i - 1], est["dir"][i]
    va, vb = est["vel"][i - 1], est["vel"][i]
    ra, rb = math.radians(da), math.radians(db)
    d = math.degrees(math.atan2((1 - w) * math.sin(ra) + w * math.sin(rb),
                                (1 - w) * math.cos(ra) + w * math.cos(rb))) % 360
    return dict(desde=d, vel=(1 - w) * va + w * vb, peso=w,
                antes=(a, float(da), float(va)), despues=(b, float(db), float(vb)))


def perfil_logaritmico(v_ref, z_ref=10.0, z=25.0, z0=0.1):
    """Extrapola la velocidad a otra altura (ley logaritmica de pared)."""
    return float(v_ref) * math.log(z / z0) / math.log(z_ref / z0)


def geometria_estacion(est, origen_utm):
    """Distancia, azimut y desnivel entre la estacion y el punto de tronadura."""
    if est.get("utm") is None or origen_utm is None:
        return None
    E, N, alt = est["utm"]
    dE, dN = E - float(origen_utm[0]), N - float(origen_utm[1])
    return dict(dist_km=math.hypot(dE, dN) / 1000.0,
                azimut=math.degrees(math.atan2(dE, dN)) % 360.0,
                desnivel_m=(float(origen_utm[2]) - alt) if (alt is not None and
                            len(origen_utm) > 2) else None)


def comparar_estacion(estaciones, ts_evento, desde_optico, vel_optica,
                      z_pluma=25.0, z_ref=10.0, z0=0.1, origen_utm=None,
                      verbose=True):
    """Contrasta la estimacion optica contra una o varias estaciones."""
    if isinstance(estaciones, dict):
        estaciones = [estaciones]
    filas = []
    for est in estaciones:
        try:
            w = viento_en(est, ts_evento)
        except ValueError as e:
            if verbose:
                print(f"  {est['nombre']}: {e}")
            continue
        z_est = est.get("z_ref") or z_ref
        esp = perfil_logaritmico(w["vel"], z_est, z_pluma, z0)
        filas.append(dict(nombre=est["nombre"], fuente=est["fuente"],
                          desde=w["desde"], vel_ref=w["vel"], vel_esperada=esp,
                          z_ref=z_est,
                          dif_dir=float((desde_optico - w["desde"] + 180) % 360 - 180),
                          razon_vel=vel_optica / esp if esp else float("nan"),
                          antes=w["antes"], despues=w["despues"],
                          geo=geometria_estacion(est, origen_utm)))
    if verbose and filas:
        print(f"  evento: {ts_evento}   |   optico: DESDE {desde_optico:.1f} deg, "
              f"{vel_optica:.2f} m/s a {z_pluma:.0f} m")
        print(f"  {'estacion':<26} {'DESDE':>8} {'dif':>7} {'v a %dm' % z_ref:>9}"
              f" {'v a %dm esp' % z_pluma:>12} {'razon':>7}")
        for f in filas:
            print(f"  {f['nombre'][:26]:<26} {f['desde']:8.1f} {f['dif_dir']:+7.1f} "
                  f"{f['vel_ref']:9.2f} {f['vel_esperada']:12.2f} {f['razon_vel']:7.2f}")
        for f in filas:
            a, b = f["antes"], f["despues"]
            print(f"    {f['nombre']}: interpolado entre {a[0]:%H:%M} "
                  f"({a[1]:.0f} deg, {a[2]:.1f} m/s) y {b[0]:%H:%M} "
                  f"({b[1]:.0f} deg, {b[2]:.1f} m/s)")

    return filas


brujula_en_imagen = F0.brujula_en_imagen


def graficar(res, fps, pivot_frame, u_px=None, v_px=None, bg=None, cam=None,
             dem=None, gsd_m_px=None, t_ini=None, t_fin=None, diag=None,
             ruta_png=None,
             titulo="Fase 3 - Centroide en coordenadas de video"):
    """Figura de la serie de viento en coordenadas de video."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    t = (np.asarray(res["frames"], float) - pivot_frame) / fps
    vel = np.asarray(res["speed_px_s"], float)
    vms = res.get("speed_m_s")
    y = np.asarray(vms) if vms is not None else (
        vel * gsd_m_px if gsd_m_px else vel)
    uni = "m/s" if (vms is not None or gsd_m_px) else "px/s"

    fig = plt.figure(figsize=(15, 6.0))
    G = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                          left=0.05, right=0.97, top=0.88, bottom=0.11)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    ax = fig.add_subplot(G[0, :2])
    if bg is not None:
        im = bg[..., ::-1] if (np.ndim(bg) == 3) else bg
        ax.imshow((np.asarray(im) * 0.66).astype(np.uint8),
                  cmap=None if np.ndim(bg) == 3 else "gray")
    if u_px is not None and v_px is not None:
        sc = ax.scatter(u_px, v_px, c=t, cmap="plasma", s=18, zorder=3)
        ax.plot(u_px, v_px, color="w", lw=1.0, alpha=.55, zorder=2)
        ax.plot(u_px[0], v_px[0], "ws", ms=10, mec="k", zorder=4)
        ax.plot(u_px[-1], v_px[-1], "w^", ms=11, mec="k", zorder=4)
        plt.colorbar(sc, ax=ax, label="t desde tronadura (s)", pad=0.01,
                     fraction=0.035)
        ax.set_title("Centroide real de la columna sobre el frame de fondo")
    if cam is not None:
        brujula_en_imagen(ax, cam, 0.66 * cam.W, 0.60 * cam.H,
                          cam.z_piso, largo_m=35.0, dem=dem,
                          sol=getattr(cam, "sol", None))
        ax.set_xlim(0, cam.W); ax.set_ylim(cam.H, 0)
    ax.set_xlabel("u (px de trabajo)"); ax.set_ylabel("v (px de trabajo)")

    ax = fig.add_subplot(G[0, 2])
    ax.plot(t, y, color="#1E88E5", lw=1.5, zorder=3)
    if t_ini is not None:
        ax.axvspan(t.min(), t_ini, color="0.90", zorder=0)
        ax.text(t.min() + 0.15, ax.get_ylim()[1] * .96,
                "energia de\ntronadura", fontsize=7, va="top")
        m = t >= t_ini if t_fin is None else (t >= t_ini) & (t <= t_fin)
        ax.axhline(float(np.mean(y[m])), color="#1E88E5", ls="--", lw=.9)
    if t_fin is not None and t_fin < t.max():
        ax.axvspan(t_fin, t.max(), color="0.90", zorder=0)
        etq = "nube\ndiluida"
        if diag and diag.get("borde", {}).get("t") is not None \
                and abs(diag["borde"]["t"] - t_fin) < 1e-6:
            etq = "sale del\ncuadro"
        ax.text(t_fin + 0.15, ax.get_ylim()[1] * .96, etq, fontsize=7, va="top")
    ax.set_xlabel("t desde tronadura (s)")
    ax.set_ylabel(f"velocidad ({uni})", color="#1E88E5")
    ax.tick_params(axis="y", colors="#1E88E5")
    ax.set_title("Velocidad aparente", fontsize=10); ax.grid(alpha=.3)

    if diag and "textura" in diag:
        ax2 = ax.twinx()
        ax2.plot(diag["textura"]["serie_t"], diag["textura"]["serie"],
                 color="0.45", lw=1.0, ls="-", zorder=1)
        ax2.axhline(diag["textura"]["umbral"], color="0.45", lw=.8, ls=":")
        ax2.axhline(0.5, color="0.70", lw=.8, ls="--")
        ax2.set_ylim(0, 0.75)
        ax2.set_ylabel("textura (0.5 = paridad con el terreno)",
                       fontsize=7, color="0.45")
        ax2.tick_params(labelsize=7, colors="0.45")

    if ruta_png:
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
        plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
        print(f"[fig] {F0.ruta_corta(ruta_png)}")
    return fig


def graficar_validacion_dem(cam, dem, bg, XM, YM, ZM, geo=None, t=None,
                            t_ini=None, t_fin=None, alt=None, escala=None,
                            ruta_ortofoto=None, estacion_desde=None,
                            estaciones_desde=None,
                            largo_m=60.0, ruta_png=None):
    """Vista en planta: la escena georreferenciada con el resultado de la fase."""
    import matplotlib.pyplot as plt

    ANCLA = (0.62, 0.54)

    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Puntos cardinales verificados contra el DEM de la mina",
                 fontsize=13, fontweight="bold")

    P0, _ext = F0.panel_planta_cardinales(ax, cam, dem, XM, YM, largo_m=largo_m,
                                          ancla=ANCLA,
                                          ruta_ortofoto=ruta_ortofoto)
    ax.set_title("")
    if getattr(cam, "sol", None):
        ax.plot([], [], color="#FF5252", lw=2.2, label="sombra (efemerides)")
    if alt is not None and cam.origen_utm is not None:
        c = alt["conf"]
        ax.plot(cam.origen_utm[0] + alt["E"][c], cam.origen_utm[1] + alt["N"][c],
                "-", color="#FF00FF", lw=2.6, zorder=7, label="columna (hull 3D)")
    if geo is not None and t is not None:
        s_ = np.ones(len(t), bool) if t_ini is None else (
            (t >= t_ini) & (t <= (t.max() if t_fin is None else t_fin)))
        hacia = media_circular_pond(geo["azimut_hacia"][s_], geo["vel_ms"][s_])
        a = np.radians(hacia)
        Q = (np.array([cam.origen_utm[0], cam.origen_utm[1]])
             if cam.origen_utm is not None else P0[:2])
        L = largo_m * 3.0
        ax.annotate("", xy=(Q[0] + np.sin(a) * L, Q[1] + np.cos(a) * L),
                    xytext=(Q[0], Q[1]), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", lw=3.4, color="#00E676",
                                    shrinkA=0, shrinkB=0))
        ax.plot(Q[0], Q[1], "*", color="#00E676", ms=15, mec="k", mew=.6, zorder=9)
        ax.plot([], [], color="#00E676", lw=3,
                label=f"viento hacia {hacia:.0f}\u00b0")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=.85)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.tick_params(labelsize=8)

    if ruta_png:
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
        plt.savefig(ruta_png, dpi=140, bbox_inches="tight")
        print(f"[fig] {F0.ruta_corta(ruta_png)}")
    return fig


def guardar_serie_geo(geo, t, ruta_csv, calidad=None):
    """CSV con el pixel medido, la cota del plano, los m/px aplicados, el UTM y el viento.
    """
    filas = []
    for i, f in enumerate(geo["frames"]):
        filas.append([int(f), f"{t[i]:.3f}", f"{geo['u_px'][i]:.1f}",
                      f"{geo['v_px'][i]:.1f}", f"{geo['z_adv'][i]:.2f}",
                      f"{geo['m_por_px'][i]:.5f}", f"{geo['vel_ms'][i]:.3f}",
                      f"{geo['azimut_hacia'][i]:.2f}", f"{geo['azimut_desde'][i]:.2f}",
                      f"{geo['E'][i]:.2f}", f"{geo['N'][i]:.2f}",
                      f"{geo['angulo_imagen'][i]:.2f}"])
    r = F0.guardar_csv(
        filas, ["frame", "t_s", "u_px", "v_px", "z_adv_m", "m_por_px", "vel_ms",
                "azimut_hacia_deg", "azimut_desde_deg", "utm_e", "utm_n",
                "angulo_imagen_deg"], ruta_csv)
    if calidad:
        with open(os.path.splitext(ruta_csv)[0] + "_calidad.json", "w",
                  encoding="utf-8") as fh:
            json.dump(calidad, fh, indent=2, ensure_ascii=False)
        print(f"  calidad: {calidad['calidad']}"
              + (f"  banderas: {', '.join(calidad['banderas'])}"
                 if calidad["banderas"] else ""))
    return r


def instante_tronadura(cfg, fps=FPS_DEF, geometria=None, procedencia=None,
                       verbose=True):
    """Hora local de la detonacion, con toda la cadena a la vista."""
    import datetime

    t0 = None; fuente_t0 = None
    if geometria:
        if geometria.get("t0_subframe") is not None:
            t0 = float(geometria["t0_subframe"]); fuente_t0 = "T0_SUBFRAME (airblast)"
        elif geometria.get("blast_frame") is not None:
            t0 = float(geometria["blast_frame"]); fuente_t0 = "blast_frame (airblast)"
    if t0 is None:
        t0 = float(cfg.pivot_frame or 0); fuente_t0 = "cfg.pivot_frame"

    t_ini = datetime.datetime.fromisoformat(cfg.ts_inicio_video)
    ts = t_ini + datetime.timedelta(seconds=t0 / float(fps))

    fuente_ts = "?"
    if procedencia:
        fuente_ts = (procedencia.get("fases", {}).get("0", {})
                     .get("procedencia", {}).get("ts_inicio_video", "?"))

    if verbose:
        print(f"  primer frame del video : {t_ini:%Y-%m-%d %H:%M:%S} "
              f"({cfg.tz_local})   <- {fuente_ts}")
        print(f"  airblast en el clip    : frame {t0:.3f} / {fps:.3f} fps "
              f"= {t0/fps:6.3f} s   <- {fuente_t0}")
        print(f"  INSTANTE DE TRONADURA  : {ts:%Y-%m-%d %H:%M:%S.%f}"[:-3]
              + f" ({cfg.tz_local})")
        if cfg.pivot_frame is not None and abs(cfg.pivot_frame - t0) > 0.5:
            print(f"    (cfg.pivot_frame = {cfg.pivot_frame}, "
                  f"{cfg.pivot_frame - t0:+.2f} frames respecto de T0: el "
                  f"pipeline redondea hacia arriba para no cortar el evento)")
        print(f"  la estacion mas cercana en el tiempo esta a "
              f"{min(abs((ts.replace(minute=0, second=0, microsecond=0) - ts).total_seconds()), abs((ts.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1) - ts).total_seconds()))/60:.0f} "
              f"min; la etiqueta horaria del portal vale +-1 h")
    return dict(ts=ts, t0_frame=t0, fuente_t0=fuente_t0,
                fuente_ts_inicio=fuente_ts, ts_inicio=t_ini)


def serie_en_ventana(est, ts, horas=6.0):
    """Recorte de la serie de una estacion alrededor de `ts`."""
    import datetime
    a = ts - datetime.timedelta(hours=horas)
    b = ts + datetime.timedelta(hours=horas)
    s = [i for i, x in enumerate(est["ts"]) if a <= x <= b]
    return ([est["ts"][i] for i in s],
            np.array([est["vel"][i] for i in s], float),
            np.array([est["dir"][i] for i in s], float))


COLOR_EST = ("#1E88E5", "#E8710A")
COLOR_OPT = "#8E44AD"
TINTA = "#37474F"


def graficar_estaciones(estaciones, ts_evento, desde_optico, vel_optica,
                        z_pluma=25.0, z_ref=10.0, z0=0.1, horas=6.0,
                        banda_vel=None, dispersion_dir=None,
                        ruta_png=None, titulo=None):
    """Serie horaria de las estaciones con la estimacion optica encima."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if isinstance(estaciones, dict):
        estaciones = [estaciones]

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14.5, 6.4))
    G = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1],
                          left=0.06, right=0.97, top=0.745, bottom=0.16,
                          wspace=0.22)
    ax1 = fig.add_subplot(G[0, 0])
    ax2 = fig.add_subplot(G[0, 1], projection="polar")

    fig.suptitle(titulo or "Viento en las estaciones", fontsize=13,
                 fontweight="bold", y=0.975)

    ax1.axvline(ts_evento, color=TINTA, lw=1.0, ls="--", zorder=2)
    ax1.grid(alpha=.25, lw=.7)
    for lado in ("top", "right"):
        ax1.spines[lado].set_visible(False)

    vmax = [vel_optica]
    for k, est in enumerate(estaciones):
        c = COLOR_EST[k % len(COLOR_EST)]
        T, V, D = serie_en_ventana(est, ts_evento, horas)
        if not T:
            continue
        ax1.plot(T, V, "-o", color=c, lw=2.0, ms=6, zorder=4)
        ax1.annotate(est["nombre"], xy=(T[-1], V[-1]), xytext=(8, 0),
                     textcoords="offset points", fontsize=9, color=TINTA,
                     va="center", annotation_clip=False)
        vmax.append(float(np.nanmax(V)))
    if banda_vel:
        ax1.errorbar([ts_evento], [vel_optica],
                     yerr=[[max(0.0, vel_optica - min(banda_vel))],
                           [max(0.0, max(banda_vel) - vel_optica)]],
                     fmt="none", ecolor=COLOR_OPT, elinewidth=2.0, capsize=5,
                     zorder=6)
    ax1.plot([ts_evento], [vel_optica], "*", color=COLOR_OPT, ms=20, zorder=7,
             mec="white", mew=1.2)
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(min(0.0, y0), y1 + 0.12 * (y1 - y0))
    ax1.annotate("tronadura", xy=(ts_evento, ax1.get_ylim()[1]), xytext=(4, -4),
                 textcoords="offset points", fontsize=8, color=TINTA,
                 va="top", ha="left")
    ax1.set_ylabel("velocidad (m/s)", color=TINTA)
    ax1.set_xlabel(f"hora local ({estaciones[0]['tz'] if estaciones else ''})",
                   color=TINTA)
    ax1.set_title("Velocidad", fontsize=11, color=TINTA, loc="left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(horas // 3))))

    ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.arange(0, 360, 45),
                       ["N", "NE", "E", "SE", "S", "SO", "O", "NO"])
    ax2.grid(alpha=.30, lw=.7)
    ax2.set_title("Direccion", fontsize=11, color=TINTA, pad=26)
    ax2.set_rlabel_position(205)

    R = max(vmax) * 1.12 if vmax else 1.0
    for k, est in enumerate(estaciones):
        c = COLOR_EST[k % len(COLOR_EST)]
        try:
            w = viento_en(est, ts_evento)
        except ValueError:
            continue
        ax2.annotate("", xy=(np.radians((w["desde"] + 180) % 360), w["vel"]),
                     xytext=(np.radians(w["desde"]), w["vel"]),
                     arrowprops=dict(arrowstyle="-|>", lw=2.6, color=c), zorder=8)
        ax2.plot([np.radians(w["desde"])], [w["vel"]], "o", color=c, ms=5,
                 zorder=8)
        ax2.text(np.radians(w["desde"]), w["vel"] * 1.10,
                 f"{w['desde']:.0f}\u00b0", color=c, fontsize=9,
                 fontweight="bold", ha="center", zorder=9)
        ax2.text(np.radians((w["desde"] + 180) % 360), w["vel"] * 1.10,
                 f"{(w['desde'] + 180) % 360:.0f}\u00b0", color=c, fontsize=9,
                 fontweight="bold", ha="center", zorder=9)

    ax2.annotate("", xy=(np.radians((desde_optico + 180) % 360), vel_optica),
                 xytext=(np.radians(desde_optico % 360), vel_optica), zorder=9,
                 arrowprops=dict(arrowstyle="-|>", lw=3.0, color=COLOR_OPT))
    ax2.plot([np.radians(desde_optico % 360)], [vel_optica], "o",
             color=COLOR_OPT, ms=6, zorder=9)
    if dispersion_dir:
        for d0 in (desde_optico, desde_optico + 180):
            th = np.radians(np.linspace(d0 - dispersion_dir,
                                        d0 + dispersion_dir, 24))
            ax2.fill_between(th, 0, R, color=COLOR_OPT, alpha=.16, zorder=2)
    ax2.text(np.radians(desde_optico % 360), vel_optica * 1.10 + 0.06 * R,
             f"{desde_optico % 360:.0f}\u00b0", color=COLOR_OPT, fontsize=10,
             fontweight="bold", ha="center", zorder=9)
    ax2.text(np.radians((desde_optico + 180) % 360),
             vel_optica * 1.10 + 0.06 * R,
             f"{(desde_optico + 180) % 360:.0f}\u00b0", color=COLOR_OPT,
             fontsize=10, fontweight="bold", ha="center", zorder=9)
    ax2.set_ylim(0, R)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=8)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    manijas = [Line2D([], [], color=COLOR_EST[k % len(COLOR_EST)], lw=2.0,
                      marker="o", ms=6, label=e["nombre"])
               for k, e in enumerate(estaciones)]
    manijas.append(Line2D([], [], color=COLOR_OPT, lw=0, marker="*", ms=16,
                          label="Flujo optico"))
    if dispersion_dir:
        manijas.append(Patch(facecolor=COLOR_OPT, alpha=.16,
                             label=f"dispersion direccional +-{dispersion_dir:.0f} deg"))
    fig.legend(handles=manijas, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               fontsize=9, frameon=False, ncol=len(manijas))

    if ruta_png:
        os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
        plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
        print(f"[fig] {F0.ruta_corta(ruta_png)}")
    return fig


ESQUEMA_REGISTRO = 1

REGISTRO_DEFECTO = {
    "esquema": ESQUEMA_REGISTRO,
    "_leeme": (
        "Una carpeta por estacion. En la de la DMC va un archivo por mes "
        "(el nombre da igual: el mes se lee de adentro; se sugiere AAAA-MM.xls). "
        "En la de SINCA van los dos CSV de la serie horaria, direccion y "
        "velocidad. Agregar una estacion es agregar una entrada aqui y su "
        "carpeta."),
    "estaciones": [
        {
            "id": "DMC_Chorrillos",
            "nombre": "Chorrillos (DMC)",
            "fuente": "DMC",
            "por_mes": True,
            "patron": ["*.xls", "*.xlsx", "*.htm", "*.html"],
            "latlon": [-22.450000, -68.910277, 2298.0],
            "z_anemometro_m": 10.0,
            "url": ("https://climatologia.meteochile.gob.cl/application/"
                    "informacion/inventarioComponentesPorEstacion/220026/28/60"),
            "descarga": ("Elige el MES en el selector y guarda el .xls en esta "
                         "carpeta. Es HTML con extension .xls: no lo abras y "
                         "vuelvas a guardar con Excel."),
        },
        {
            "id": "SINCA_PVergaraKeller",
            "nombre": "P.Vergara Keller (SINCA)",
            "fuente": "SINCA",
            "por_mes": False,
            "patron_dir": ["*DIR*.csv"],
            "patron_vel": ["*VEL*.csv"],
            "utm": [506893.0, 7518227.0, 2260.0],
            "z_anemometro_m": 10.0,
            "url": "https://sinca.mma.gob.cl/",
            "descarga": ("Serie de tiempo horaria (NO la rosa de vientos), "
                         "direccion y velocidad por separado. El rango va en el "
                         "nombre; para extenderlo hay que volver a exportar."),
        },
    ],
}


def dir_estaciones(dir_datos) -> str:
    return os.path.join(dir_datos, "estaciones")


def registro_estaciones(dir_datos, crear=True, verbose=True) -> dict:
    """El registro de estaciones. Lo crea con los valores por defecto si falta."""
    d = dir_estaciones(dir_datos)
    ruta = os.path.join(d, "estaciones.json")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
        if reg.get("esquema") != ESQUEMA_REGISTRO and verbose:
            print(f"  [aviso] {os.path.basename(ruta)} es del esquema "
                  f"{reg.get('esquema')} y este codigo espera "
                  f"{ESQUEMA_REGISTRO}.")
        return reg
    if not crear:
        return dict(REGISTRO_DEFECTO)
    os.makedirs(d, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(REGISTRO_DEFECTO, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  [estaciones] registro creado: {F0.ruta_corta(ruta)}")
    return dict(REGISTRO_DEFECTO)


def _glob_varios(carpeta, patrones):
    """Los archivos de `carpeta` que calzan con cualquiera de los patrones."""
    import glob
    if isinstance(patrones, str):
        patrones = [patrones]
    r = []
    for p in patrones:
        for v in (p, p.upper(), p.lower()):
            r += glob.glob(os.path.join(carpeta, v))
    return sorted(set(r))


def buscar_estaciones(dir_datos, tz_local="America/Santiago", verbose=True):
    """Los archivos de cada estacion del registro. Devuelve una lista de dicts."""
    reg = registro_estaciones(dir_datos, verbose=verbose)
    base = dir_estaciones(dir_datos)
    salida = []
    for e in reg.get("estaciones", []):
        e = dict(e)
        carpeta = os.path.join(base, e.get("carpeta", e["id"]))
        e["carpeta_ruta"] = carpeta
        if e["fuente"].upper() == "SINCA":
            pd_, pv = e.get("patron_dir", ["*DIR*.csv"]), e.get("patron_vel", ["*VEL*.csv"])
            arch_dir, arch_vel = _glob_varios(carpeta, pd_), _glob_varios(carpeta, pv)
            e["origen"] = "carpeta"
            if not (arch_dir and arch_vel):
                sueltos_d, sueltos_v = _glob_varios(dir_datos, pd_), _glob_varios(dir_datos, pv)
                if sueltos_d and sueltos_v:
                    arch_dir, arch_vel, e["origen"] = sueltos_d, sueltos_v, "datos/ (suelto)"
            arch_dir = [p for p in arch_dir if os.path.getsize(p) > 50_000] or arch_dir
            arch_vel = [p for p in arch_vel if os.path.getsize(p) > 50_000] or arch_vel
            _rec = lambda L: sorted(L, key=os.path.getmtime, reverse=True)
            e["archivos_dir"], e["archivos_vel"] = _rec(arch_dir), _rec(arch_vel)
            e["archivos"] = e["archivos_dir"][:1] + e["archivos_vel"][:1]
        else:
            pats = e.get("patron", ["*.xls", "*.xlsx", "*.htm", "*.html"])
            arch = _glob_varios(carpeta, pats)
            e["origen"] = "carpeta"
            if not arch:
                arch = _glob_varios(dir_datos, [f"*{e['id'].split('_')[0]}*{p[1:]}"
                                                for p in pats])
                if arch:
                    e["origen"] = "datos/ (suelto)"
            e["archivos"] = sorted(arch)
        salida.append(e)

    if verbose:
        for e in salida:
            n = len(e["archivos"])
            if not n:
                print(f"  {e['id']:24s}: SIN ARCHIVOS  ({F0.ruta_corta(e['carpeta_ruta'])})")
                continue
            det = ", ".join(os.path.basename(a) for a in e["archivos"][:4])
            if n > 4:
                det += f", +{n - 4} mas"
            print(f"  {e['id']:24s}: {n} archivo(s) [{e['origen']}]  {det}")
        if any(e["origen"] == "datos/ (suelto)" for e in salida):
            print(f"    (hay archivos sueltos en datos/. Lo ordenado es "
                  f"{base}{os.sep}<estacion>{os.sep}; F3.ordenar_estaciones() los mueve.)")
    return salida


def _mes(ts):
    return f"{ts.year:04d}-{ts.month:02d}"


def cobertura_estacion(est, ts_evento, margen_h=6.0):
    """Si la serie de `est` cubre `ts_evento` con `margen_h` a cada lado."""
    import datetime
    T = est.get("ts") or []
    if not T:
        return dict(ok=False, motivo="serie vacia", mes=_mes(ts_evento))
    ini, fin = T[0], T[-1]
    if ts_evento < ini:
        return dict(ok=False, motivo="el evento es ANTERIOR a la serie",
                    mes=_mes(ts_evento), rango=(ini, fin))
    if ts_evento > fin:
        return dict(ok=False, motivo="el evento es POSTERIOR a la serie",
                    mes=_mes(ts_evento), rango=(ini, fin))
    ventana = datetime.timedelta(hours=margen_h)
    cerca = [t for t in T if abs((t - ts_evento).total_seconds()) <= ventana.total_seconds()]
    if not cerca:
        return dict(ok=False, motivo=f"hueco de mas de {margen_h:.0f} h "
                                     f"alrededor del evento",
                    mes=_mes(ts_evento), rango=(ini, fin))
    d_h = min(abs((t - ts_evento).total_seconds()) for t in cerca) / 3600.0
    parcial = len(cerca) < margen_h
    return dict(ok=True, parcial=bool(parcial), rango=(ini, fin),
                n_ventana=len(cerca), dist_h=d_h, mes=_mes(ts_evento))


def informe_cobertura(estaciones, ts_evento, sin_archivos=(), margen_h=6.0,
                      verbose=True):
    """Aviso GRANDE de lo que falta para comparar este evento, y de donde sale."""
    faltan = []
    for e in sin_archivos:
        faltan.append(dict(id=e.get("id"), nombre=e.get("nombre", e.get("id")),
                           motivo=e.get("motivo") or
                                  "no hay ningun archivo en su carpeta",
                           mes=_mes(ts_evento), url=e.get("url"),
                           descarga=e.get("descarga"),
                           carpeta=e.get("carpeta_ruta"), por_mes=e.get("por_mes")))
    for est in estaciones:
        c = cobertura_estacion(est, ts_evento, margen_h)
        est["cobertura"] = c
        if not c["ok"]:
            faltan.append(dict(id=est.get("id"), nombre=est["nombre"],
                               motivo=c["motivo"], mes=c["mes"],
                               rango=c.get("rango"), url=est.get("url"),
                               descarga=est.get("descarga"),
                               carpeta=est.get("carpeta_ruta"),
                               por_mes=est.get("por_mes"),
                               archivos=est.get("archivos"),
                               meses=_meses_dmc(est, verbose=False)))
    if verbose and faltan:
        print("\n" + "!" * 74)
        print(f"!!  FALTAN DATOS DE ESTACION PARA {ts_evento:%Y-%m-%d %H:%M} "
              f"(mes {_mes(ts_evento)})")
        print("!" * 74)
        for f in faltan:
            print(f"!!  {f['nombre']}: {f['motivo']}")
            if f.get("rango"):
                print(f"!!      la serie que hay va de {f['rango'][0]:%Y-%m-%d} "
                      f"a {f['rango'][1]:%Y-%m-%d}")
            if f.get("por_mes"):
                print(f"!!      hace falta el mes {f['mes']}")
                if f.get("archivos"):
                    print(f"!!      archivos en la carpeta: "
                          f"{', '.join(f['archivos'])}")
                if f.get("meses"):
                    print(f"!!      meses que traen POR DENTRO: "
                          f"{', '.join(f['meses'])}")
            if f.get("url"):
                print(f"!!      bajar de: {f['url']}")
            if f.get("descarga"):
                print(f"!!      {f['descarga']}")
            if f.get("carpeta"):
                print(f"!!      dejarlo en: {f['carpeta']}")
        print("!!  La comparacion sigue con las estaciones que SI cubren el "
              "evento,")
    elif verbose:
        print(f"  cobertura: las {len(estaciones)} estacion(es) cubren "
              f"{ts_evento:%Y-%m-%d %H:%M}")
        for est in estaciones:
            c = est.get("cobertura", {})
            if c.get("parcial"):
                print(f"    [aviso] {est['nombre']} solo trae "
                      f"{c['n_ventana']} muestras en +-{margen_h:.0f} h")
    return faltan


def cargar_estaciones(dir_datos, tz_local="America/Santiago", ts_evento=None,
                      margen_h=6.0, verbose=True, **compat):
    """Todas las estaciones del registro que tengan datos en `datos/estaciones/`."""
    for k in compat:
        if verbose:
            print(f"  [aviso] cargar_estaciones ya no usa '{k}': eso vive ahora "
                  f"en {F0.ruta_corta(os.path.join(dir_estaciones(dir_datos), 'estaciones.json'))}")

    entradas = buscar_estaciones(dir_datos, tz_local, verbose=verbose)
    est, sin_archivos = [], []
    for e in entradas:
        if not e["archivos"]:
            sin_archivos.append(e)
            continue
        coords = dict(coords_latlon=e.get("latlon"), coords_utm=e.get("utm"))
        try:
            if e["fuente"].upper() == "SINCA":
                if not (e["archivos_dir"] and e["archivos_vel"]):
                    que = "velocidad (*VEL*.csv)" if e["archivos_dir"] \
                          else "direccion (*DIR*.csv)"
                    e["motivo"] = f"falta el CSV de {que}; sin los dos no carga"
                    print(f"  [aviso] {e['id']}: {e['motivo']}")
                    sin_archivos.append(e)
                    continue
                s = cargar_estacion_sinca(e["archivos_dir"][0], e["archivos_vel"][0],
                                          nombre=e["nombre"], verbose=verbose,
                                          **coords)
            else:
                s = cargar_estacion_dmc(e["archivos"], tz_local=tz_local,
                                        nombre=e["nombre"], verbose=verbose,
                                        **coords)
        except Exception as ex:
            e["motivo"] = f"sus archivos no se pudieron leer ({type(ex).__name__}: {ex})"
            print(f"  [ERROR] {e['id']}: {type(ex).__name__}: {ex}")
            sin_archivos.append(e)
            continue
        s["id"] = e["id"]
        s.setdefault("archivos", [os.path.basename(a) for a in e["archivos"]])
        s["url"] = e.get("url")
        s["descarga"] = e.get("descarga")
        s["por_mes"] = e.get("por_mes")
        s["carpeta_ruta"] = e.get("carpeta_ruta")
        s["z_ref"] = e.get("z_anemometro_m")
        est.append(s)

    if verbose:
        print(f"  -> {len(est)} estacion(es) cargada(s)")
    if ts_evento is not None:
        informe_cobertura(est, ts_evento, sin_archivos=sin_archivos,
                          margen_h=margen_h, verbose=verbose)
    elif sin_archivos and verbose:
        for e in sin_archivos:
            print(f"  [aviso] {e['id']} no tiene archivos en {F0.ruta_corta(e['carpeta_ruta'])}")
    return est


def desfase_a_azimut(res, geo, m=None):
    """Relacion entre el angulo en la imagen y el azimut geografico, ajustada sobre la
    serie.
    """
    a = np.asarray(res["angle_deg"], float)
    az = np.asarray(geo["azimut_hacia"], float)
    if m is not None:
        a, az = a[m], az[m]
    cand = [(dispersion_circular_deg(az - s * a), s,
             media_circular_deg(az - s * a) % 360.0) for s in (1.0, -1.0)]
    disp, s, K = min(cand)
    return float(K), float(s), float(disp)


def graficar_von_mises(res, t, t_ini=None, t_fin=None, ruta_png=None,
                       n_flechas=20000, r_circulo=0.68,
                       cobertura=1, margen_curva_deg=30.0,
                       mostrar_kde=False, kappa_kde=180.0,
                       flechas_diametrales=True, geo=None, disp_max_deg=2.0):
    """Rosa de las direcciones del campo dentro de la ventana."""
    import matplotlib.pyplot as plt

    t = np.asarray(t, float)
    if t_ini is None:
        m = np.ones_like(t, bool)
    else:
        m = (t >= t_ini) & (t <= (np.inf if t_fin is None else t_fin))

    if geo is not None:
        K, s_esp, disp_K = desfase_a_azimut(res, geo, m)
        if disp_K > disp_max_deg:
            print(f"  [aviso] la relacion imagen->azimut no es rigida "
                  f"(residuo {disp_K:.1f} deg): la rosa girada es aproximada")
    else:
        K, s_esp = None, 1.0

    def _az(a):
        a = np.asarray(a, float)
        return a if K is None else np.radians(s_esp * np.degrees(a) + K)

    ang = np.asarray(res["vmf_muestra"], float)[m].ravel()
    ang = ang[np.isfinite(ang)]
    C = float(np.cos(ang).mean()); S = float(np.sin(ang).mean())
    mu = float(np.arctan2(S, C))
    Rbar = float(min(np.hypot(C, S), 1.0 - 1e-12))
    kappa = float(Rbar * (2.0 - Rbar ** 2) / (1.0 - Rbar ** 2))
    sigma = float(np.degrees(kappa ** -0.5)) if kappa > 0 else np.nan

    d = (ang - mu + np.pi) % (2 * np.pi) - np.pi
    q = 100.0 * (1.0 - cobertura) / 2.0
    lo, hi = np.percentile(d, [q, 100.0 - q])
    dentro = (d >= lo) & (d <= hi)

    mrg = np.radians(margen_curva_deg)
    aa = mu + np.linspace(max(lo - mrg, -np.pi), min(hi + mrg, np.pi), 601)
    vm = np.exp(kappa * np.cos(aa - mu)) / (2 * np.pi * np.i0(kappa))
    if mostrar_kde:
        kde = np.array([np.exp(kappa_kde * np.cos(x - ang)).sum() for x in aa])
        kde /= ang.size * 2 * np.pi * np.i0(kappa_kde)
    else:
        kde = None
    esc = (1.0 - r_circulo) / max(float(vm.max()),
                                  float(kde.max()) if kde is not None else 0.0)

    fig = plt.figure(figsize=(6.6, 6.2))
    axp = fig.add_subplot(1, 1, 1, projection="polar")
    axp.set_theta_direction(-1)
    if K is None:
        axp.set_theta_zero_location("E")
    else:
        axp.set_theta_zero_location("N")
        axp.set_thetagrids(np.arange(0, 360, 45),
                           ["N", "NE", "E", "SE", "S", "SO", "O", "NO"])

    sub = ang[dentro]
    sub = sub[:: max(1, sub.size // n_flechas)]
    r_flecha = 0.88 * r_circulo
    for a_i in sub:
        cola = (_az(a_i + np.pi), r_flecha) if flechas_diametrales else (0.0, 0.0)
        axp.annotate("", xy=(_az(a_i), r_flecha), xytext=cola,
                     arrowprops=dict(arrowstyle="-|>", color="C0", lw=0.9,
                                     alpha=0.75, shrinkA=0, shrinkB=0,
                                     mutation_scale=8),
                     zorder=2)
    aa0 = np.linspace(-np.pi, np.pi, 361)
    axp.plot(aa0, np.full_like(aa0, r_circulo), color="0.35", lw=1.0, zorder=5)

    if kde is not None:
        axp.plot(_az(aa), r_circulo + esc * kde, color="0.25", lw=1.4, zorder=6,
                 label="KDE circular (sigue los datos)")
    axp.plot(_az(aa), r_circulo + esc * vm, color="C3", lw=2.0, zorder=7,
             label=fr"von Mises ajustada ($\kappa$={kappa:.1f})")
    r_mu = r_circulo + esc * float(vm.max())
    mu_pl = float(_az(mu))
    mu_gr = np.degrees(mu_pl) % 360.0
    axp.annotate("", xy=(mu_pl, r_mu),
                 xytext=((_az(mu + np.pi), r_mu) if flechas_diametrales
                         else (0.0, 0.0)),
                 arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.8,
                                 shrinkA=0, shrinkB=0), zorder=8)
    caja = dict(fc="w", ec="none", alpha=.75, boxstyle="round,pad=0.15")
    axp.text(mu_pl, r_mu * 1.14, f"hacia {mu_gr:.0f}\u00b0",
             color="C3", fontsize=9, fontweight="bold", ha="center", zorder=9,
             bbox=caja)
    if flechas_diametrales:
        axp.text(_az(mu + np.pi), r_mu * 1.14,
                 f"desde {(mu_gr + 180) % 360:.0f}\u00b0",
                 color="C3", fontsize=9, fontweight="bold", ha="center",
                 zorder=9, bbox=caja)
    axp.plot([], [], color="C0", lw=0.7, alpha=0.7,
             label=f"direcciones ({sub.size} de {ang.size} flechas)")

    axp.set_ylim(0, 1.03)
    axp.set_yticklabels([])
    axp.yaxis.grid(False)
    axp.xaxis.grid(alpha=0.12)
    encabezado = ("Direcciones del campo en plano imagen" if K is None
                  else "Direcciones del campo en azimut geografico")
    axp.set_title(encabezado + "\n"
                  + fr"$\mu$ = {mu_gr:.1f}$\degree$,  "
                  + fr"$\bar R$ = {Rbar:.3f},  "
                  + fr"$\sigma \approx$ {sigma:.1f}$\degree$", pad=14)
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=1,
               fontsize=8, frameon=False)

    if ruta_png:
        fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
        print(f"  -> {F0.ruta_corta(ruta_png)}")
    return fig, dict(mu_deg=float(np.degrees(mu)), kappa=kappa, Rbar=Rbar,
                     sigma_deg=sigma,
                     mu_azimut_deg=(float(mu_gr) if K is not None else None))
