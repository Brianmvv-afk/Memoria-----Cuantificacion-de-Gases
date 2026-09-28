"""Fase 4: leyes de dispersion medidas sobre la mascara y concentracion normalizada chi =
C/Q del puff gaussiano con reflexion en el suelo.
"""

import json
import os

import numpy as np

import funciones_F0 as F0


CRITERIOS_DEFECTO = dict(borde_max=0.01,
                         cob_max=0.35,
                         area_min=500,
                         salto_max=0.60)


def metricas_por_frame(masks, area_min=500):
    met = {}
    for fi, m in masks.items():
        r = F0.metricas_mascara(m, area_min)
        if r is not None:
            met[fi] = r
    return met


def ventana_valida(met, criterios=None):
    """Mayor tramo contiguo que cumple los criterios geometricos."""
    c = dict(CRITERIOS_DEFECTO, **(criterios or {}))
    fs = sorted(met)
    ok, area_prev = {}, None
    for fi in fs:
        r = met[fi]
        c_salto = True
        if area_prev:
            c_salto = abs(r["area"] - area_prev) / area_prev <= c["salto_max"]
        ok[fi] = (r["frac_borde"] <= c["borde_max"]
                  and r["cobertura"] <= c["cob_max"] and c_salto)
        area_prev = r["area"]
    mejor, actual = [], []
    for fi in fs:
        if ok[fi]:
            actual.append(fi)
            if len(actual) > len(mejor):
                mejor = list(actual)
        else:
            actual = []
    return mejor, ok


_SQRT2PI3 = (2.0 * np.pi) ** 1.5


def pesos_opacidad(frame_gris, fondo_gris, mask):
    """Peso por pixel proporcional al contraste contra el fondo."""
    m = np.asarray(mask) > 0
    d = np.abs(np.asarray(frame_gris, float) - np.asarray(fondo_gris, float))
    w = np.where(m, d, 0.0)
    return w


def sigmas_mundo(masks, cam, frames, z_adv, pesos=None, max_px=20000,
                 corregir_area=True, semilla=0, verbose=True):
    """Momentos de la nube MEDIDOS EN METROS sobre el plano de adveccion."""
    rng = np.random.default_rng(semilla)
    z_adv = np.atleast_1d(np.asarray(z_adv, float))
    if len(z_adv) == 1:
        z_adv = np.repeat(z_adv, len(frames))
    if pesos is None and verbose:
        print("  [aviso] sin pesos de opacidad: los momentos son de la "
              "SILUETA binaria y sobreestiman sigma (+50 % tipico). "
              "Pasa `pesos` para medir la distribucion, no el contorno.")
    out = []
    for k, fi in enumerate(frames):
        m = np.asarray(masks[fi]) > 0
        ys, xs = np.nonzero(m)
        n = len(xs)
        if n < 50:
            continue
        if n > max_px:
            idx = rng.choice(n, max_px, replace=False)
            xs, ys = xs[idx], ys[idx]
        xf, yf = xs.astype(float), ys.astype(float)
        P = cam.a_plano(xf, yf, z_adv[k])
        w = np.ones(len(xf))
        if pesos is not None:
            W = np.asarray(pesos[fi], float)
            w = W[ys, xs]
        if corregir_area:
            Pu = cam.a_plano(xf + 0.5, yf, z_adv[k])
            Pv = cam.a_plano(xf, yf + 0.5, z_adv[k])
            du = (Pu - P)[:, :2] * 2.0
            dv = (Pv - P)[:, :2] * 2.0
            area = np.abs(du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0])
            w = w * area
        ok = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1]) & np.isfinite(w) & (w > 0)
        if ok.sum() < 50 or w[ok].sum() <= 0:
            continue
        E, N, ww = P[ok, 0], P[ok, 1], w[ok]
        ww = ww / ww.sum()
        cen = np.array([float(E @ ww), float(N @ ww)])
        dE, dN = E - cen[0], N - cen[1]
        S = np.array([[float((ww * dE) @ dE), float((ww * dE) @ dN)],
                      [float((ww * dE) @ dN), float((ww * dN) @ dN)]])
        ev, evec = np.linalg.eigh(S)
        out.append(dict(frame=int(fi), n_px=int(n), n_usados=int(ok.sum()),
                        centro=cen, cov=S, z=float(z_adv[k]),
                        sig_may=float(np.sqrt(max(ev[1], 0.0))),
                        sig_men=float(np.sqrt(max(ev[0], 0.0))),
                        az_may=float(np.degrees(np.arctan2(evec[0, 1],
                                                           evec[1, 1])) % 180.0),
                        pesado=pesos is not None))
    if verbose and out:
        print(f"  [sigmas] {len(out)} frames | sigma mayor "
              f"{out[0]['sig_may']:.1f} -> {out[-1]['sig_may']:.1f} m, "
              f"menor {out[0]['sig_men']:.1f} -> {out[-1]['sig_men']:.1f} m"
              f"   [{'pesado por opacidad' if pesos is not None else 'SILUETA binaria'}]")
    return out


def _sigma_en_direccion(S, az_deg):
    """Desviacion de la nube proyectada sobre un azimut (0=N, horario)."""
    a = np.radians(az_deg)
    d = np.array([np.sin(a), np.cos(a)])
    return float(np.sqrt(max(d @ S @ d, 0.0)))


def descomponer_sigmas(sig, azimut_hacia, cam, sigma_z=None, verbose=True):
    """Covarianza en el suelo -> marco del viento, con la inflacion por la altura de la
    nube.
    """
    d = cam.R[:, 2]
    az_cam = float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)
    theta = float(np.degrees(np.arccos(min(1.0, max(-1.0, -d[2])))))
    tan_t = float(np.tan(np.radians(theta)))
    phi = abs((azimut_hacia - az_cam + 90.0) % 180.0 - 90.0)

    sz_arr = None
    if sigma_z is not None:
        sz_arr = np.broadcast_to(np.atleast_1d(np.asarray(sigma_z, float)),
                                 (len(sig),))
    filas = []
    for i, s in enumerate(sig):
        S = s["cov"]
        s_al = _sigma_en_direccion(S, azimut_hacia)
        s_cr = _sigma_en_direccion(S, azimut_hacia + 90.0)
        s_cam = _sigma_en_direccion(S, az_cam)
        infl_al = abs(np.cos(np.radians(phi)))
        infl_cr = abs(np.sin(np.radians(phi)))
        if sz_arr is not None:
            corr = (float(sz_arr[i]) * tan_t) ** 2
            s_al = float(np.sqrt(max(s_al ** 2 - corr * infl_al ** 2, 1e-6)))
            s_cr = float(np.sqrt(max(s_cr ** 2 - corr * infl_cr ** 2, 1e-6)))
        filas.append(dict(frame=s["frame"], centro=s["centro"], z=s["z"],
                          sig_along=s_al, sig_cross=s_cr, sig_cam=s_cam,
                          sig_may=s["sig_may"], sig_men=s["sig_men"]))
    info = dict(az_camara=az_cam, theta_nadir=theta, tan_theta=tan_t,
                phi_vista_viento=phi, corregido=sz_arr is not None,
                sigma_z=None if sz_arr is None else float(np.mean(sz_arr)),
                infl_along=float(abs(np.cos(np.radians(phi)))),
                infl_cross=float(abs(np.sin(np.radians(phi)))))
    if verbose:
        print(f"  [proyeccion] camara al azimut {az_cam:.1f} deg, "
              f"{theta:.1f} deg del nadir (tan {tan_t:.2f})")
        print(f"               angulo vista-viento: {phi:.1f} deg  "
              f"({'perpendicular: sigma_along limpio' if phi > 60 else 'paralelo: sigma_along inflado' if phi < 30 else 'intermedio: los dos algo inflados'})")
        if sz_arr is None:
            print(f"               sigma_z NO disponible (falta la Fase 2): "
                  f"la componente paralela a la vista queda INFLADA por la "
                  f"altura de la nube y no se corrige.")
        else:
            szm = float(np.mean(sz_arr))
            print(f"               sigma_z = {szm:.1f} m medio (Fase 2): se "
                  f"descuenta hasta {(szm*tan_t):.1f} m del eje de la vista.")
    return filas, info


def ley_sigma(t, sig, modo="medido+fickiano", modelo="sigma0", t_min=None,
              verbose=True):
    """Ajusta la ley de crecimiento y devuelve una sigma(t) EXTRAPOLABLE."""
    t = np.asarray(t, float)
    s = np.asarray(sig, float)
    ok = np.isfinite(t) & np.isfinite(s) & (t > 0) & (s > 0)
    if t_min is not None:
        ok &= t >= float(t_min)
    t, s = t[ok], s[ok]
    if len(t) < 4:
        raise ValueError(f"Muy pocos puntos ({len(t)}) para ajustar sigma(t).")
    s2 = s ** 2

    if modelo == "potencia":
        alpha, logK = np.polyfit(np.log(t), np.log(s2), 1)
        alpha, K, s0_2 = float(alpha), float(np.exp(logK)), 0.0
        pred = K * t ** alpha
        r2 = 1 - np.sum((np.log(s2) - np.log(pred)) ** 2) \
            / np.sum((np.log(s2) - np.log(s2).mean()) ** 2)
    else:
        mejor = None
        for a in np.arange(0.20, 3.0001, 0.005):
            ta = t ** a
            A = np.column_stack([np.ones_like(t), ta])
            coef = np.linalg.lstsq(A, s2, rcond=None)[0]
            if coef[0] < 0.0:
                c1 = float(ta @ s2 / (ta @ ta))
                coef = np.array([0.0, c1])
            if coef[1] <= 0.0:
                continue
            res = s2 - (coef[0] + coef[1] * ta)
            sse = float(res @ res)
            if mejor is None or sse < mejor[0]:
                mejor = (sse, float(a), float(coef[0]), float(coef[1]))
        if mejor is None:
            raise ValueError("No hubo ningun alpha con K > 0 en la grilla.")
        sse, alpha, s0_2, K = mejor
        pred = s0_2 + K * t ** alpha
        r2 = 1 - sse / float(np.sum((s2 - s2.mean()) ** 2))

    t_fin = float(t.max())
    s2_fin = s0_2 + K * t_fin ** alpha
    dsdt_fin = K * alpha * t_fin ** (alpha - 1.0)
    D_fin = dsdt_fin / 2.0

    def sigma(tt):
        tt = np.maximum(np.asarray(tt, float), 0.0)
        pot = s0_2 + K * tt ** alpha
        if modo == "potencia":
            return np.sqrt(np.maximum(pot, 1e-6))
        recta = s2_fin + dsdt_fin * (tt - t_fin)
        return np.sqrt(np.maximum(np.where(tt <= t_fin, pot, recta), 1e-6))

    out = dict(alpha=float(alpha), K=K, sigma0_m=float(np.sqrt(s0_2)),
               r2=float(r2), modo=modo, modelo=modelo,
               t_min=None if t_min is None else float(t_min),
               t_ini=float(t.min()), t_fin=t_fin, n=int(len(t)),
               sigma_fin=float(np.sqrt(s2_fin)), D_fin_m2s=float(D_fin),
               sigma=sigma)
    out["regimen"] = ("fickiano" if alpha < 1.3 else
                      "superdifusivo" if alpha < 2.3 else "Richardson")
    if verbose:
        print(f"    sigma^2 = {s0_2:.3g} + {K:.3g} * t^{alpha:.2f}   "
              f"(sigma_0 = {out['sigma0_m']:.1f} m, R2 {r2:.3f}, "
              f"{out['regimen']}, n={len(t)}, medido {t.min():.2f}-{t_fin:.1f} s)")
        print(f"    sigma({t_fin:.1f} s) = {out['sigma_fin']:.1f} m | "
              f"D equivalente al final = {D_fin:.1f} m2/s")
        if modo == "medido+fickiano":
            print(f"    despues de {t_fin:.1f} s se extrapola FICKIANO "
                  f"(alpha=1), no t^{alpha:.2f}")
    return out


def chi_puff(P, t, fuente, u_ms, azimut_hacia, sig_along, sig_cross, sig_z,
             H, reflexion=True):
    """chi = C/Q [1/m3] del puff gaussiano en los puntos P (N,3 UTM) y tiempo t."""
    P = np.atleast_2d(np.asarray(P, float))
    sa = float(sig_along(t)); sc = float(sig_cross(t)); sz = float(sig_z(t))
    H = float(H(t)) if callable(H) else float(H)
    a = np.radians(float(azimut_hacia))
    e_al = np.array([np.sin(a), np.cos(a)])
    e_cr = np.array([np.cos(a), -np.sin(a)])

    d = P[:, :2] - np.asarray(fuente[:2], float)[None, :]
    x = d @ e_al - float(u_ms) * float(t)
    y = d @ e_cr
    z = P[:, 2] - float(fuente[2])

    g = np.exp(-0.5 * (x / sa) ** 2) * np.exp(-0.5 * (y / sc) ** 2)
    gz = np.exp(-0.5 * ((z - H) / sz) ** 2)
    if reflexion:
        gz = gz + np.exp(-0.5 * ((z + H) / sz) ** 2)
    return g * gz / (_SQRT2PI3 * sa * sc * sz)


def _altura(par):
    """Altura para evaluar chi: la ley H(t) de la Fase 2 si existe, si no el escalar."""
    ley = par.get("ley_H")
    return ley if callable(ley) else float(par["H"])


def _altura_en(par, t):
    """`_altura` evaluada en un instante."""
    ley = par.get("ley_H")
    return float(ley(t)) if callable(ley) else float(par["H"])


def _leer_viento_georref(ruta):
    """viento_georreferenciado.csv -> dict de arrays."""
    import csv
    col = {}
    with open(ruta, "r", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            for k, v in fila.items():
                try:
                    col.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    col.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in col.items()}


def preparar_dispersion(cfg, masks, cam, fps, geometria, ventana=None,
                        sigma_z_ley=None, modo_extrapolacion="medido+fickiano",
                        modelo_sigma="sigma0", t_min_ajuste=None,
                        usar_opacidad=True, exigir_calidad=True, verbose=True):
    """Reune todo lo que el modelo necesita, y se planta si el insumo no da."""
    ruta_cal = cfg.ruta(3, "viento_georreferenciado_calidad.json")
    calidad = {}
    if os.path.exists(ruta_cal):
        with open(ruta_cal, "r", encoding="utf-8") as fh:
            calidad = json.load(fh)
    ruta_geo = cfg.ruta(3, "viento_georreferenciado.csv")
    if not os.path.exists(ruta_geo):
        raise FileNotFoundError(
            f"Falta {ruta_geo}. Este bloque transporta con el viento medido en "
            f"la Fase 3: corre la Fase 3 antes.")
    W = _leer_viento_georref(ruta_geo)

    ruta_mundo = cfg.ruta(3, "viento_mundo.csv")
    u_ms = az_hacia = None
    if os.path.exists(ruta_mundo):
        import csv as _csv
        with open(ruta_mundo, "r", encoding="utf-8") as fh:
            filas = list(_csv.DictReader(fh))
        prom = [f for f in filas if str(f["frame"]).startswith("PROMEDIO")]
        if prom:
            u_ms = float(prom[-1]["vel_ms"])
            az_hacia = (float(prom[-1]["dir_desde_deg"]) + 180.0) % 360.0

    if verbose:
        print(f"  calidad del viento (Fase 3): {calidad.get('calidad', '?')}")
        for b in calidad.get("banderas", []):
            print(f"      - {b}")
    if calidad.get("calidad") == "dudoso":
        msg = ("El viento de la Fase 3 esta marcado como DUDOSO "
               f"({', '.join(calidad.get('banderas', []))}).\n"
               "    Este bloque ADVECTA con ese viento: si la direccion esta "
               "mal, el mapa entero apunta a otro lado y se ve igual de bien.\n"
               "    Corrige la Fase 3, o pasa exigir_calidad=False y trata el "
               "resultado como una demostracion del metodo, NO como un "
               "resultado del evento.")
        if exigir_calidad:
            raise RuntimeError(msg)
        print("\n" + "!" * 74 + f"\n!!  {msg}\n" + "!" * 74 + "\n")

    frames = np.asarray(W["frame"], int)
    if ventana is None:
        ventana = [int(f) for f in frames if f in masks]
    ventana = [int(f) for f in ventana if f in masks]
    if len(ventana) < 8:
        raise ValueError(f"Ventana muy corta ({len(ventana)} frames) para "
                         f"ajustar sigma(t).")

    idx = {int(f): i for i, f in enumerate(frames)}
    z_adv = np.array([W["z_adv_m"][idx[f]] if f in idx else np.nan
                      for f in ventana], float)
    z_adv = np.where(np.isfinite(z_adv), z_adv, np.nanmedian(z_adv))
    z_piso = float(cam.z_piso) if cam.z_piso is not None else float(np.nanmin(z_adv))
    H = float(np.nanmean(z_adv) - z_piso)

    pesos = None
    if usar_opacidad:
        try:
            import cv2
            bg_gray = F0.cargar_fondo(cfg.art_fondo)[0]
            pesos = {}
            with F0.LectorVideo(cfg.video, cfg.escala) as lec:
                for fi, img in lec.recorrer(ventana):
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    m = F0.normalizar_mascara(masks[fi], g.shape) > 0
                    pesos[fi] = pesos_opacidad(g, bg_gray, m)
            ventana = [f for f in ventana if f in pesos]
            if verbose:
                print(f"  [opacidad] pesos de {len(pesos)} frames "
                      f"(contraste contra modelo_fondo.npz)")
        except Exception as ex:
            print(f"  [aviso] no pude calcular los pesos de opacidad "
                  f"({type(ex).__name__}: {ex}). Sigo con la silueta binaria, "
                  f"que sobreestima sigma.")
            pesos = None
    z_adv = np.array([W["z_adv_m"][idx[f]] if f in idx else np.nan
                      for f in ventana], float)
    z_adv = np.where(np.isfinite(z_adv), z_adv, np.nanmedian(z_adv))
    sig = sigmas_mundo(masks, cam, ventana, z_adv, pesos=pesos, verbose=verbose)
    if len(sig) < 8:
        raise ValueError(f"Solo {len(sig)} frames dieron momentos utiles.")
    if u_ms is None or az_hacia is None:
        m = np.isfinite(W["vel_ms"])
        u_ms = float(np.nanmedian(W["vel_ms"][m]))
        az_hacia = float(np.nanmedian(W["azimut_hacia_deg"][m]))
    t0 = float(geometria.get("t0_subframe", cfg.pivot_frame))
    t_sig = np.array([(s["frame"] - t0) / float(fps) for s in sig])
    sz = sigma_z_ley
    if isinstance(sz, dict) and callable(sz.get("sigma")):
        sz = np.asarray(sz["sigma"](t_sig), float)
    filas, proj = descomponer_sigmas(sig, az_hacia, cam, sigma_z=sz,
                                     verbose=verbose)

    t = np.array([(f["frame"] - t0) / float(fps) for f in filas])
    if verbose:
        print("  ley de crecimiento a lo largo del viento:")
    ley_al = ley_sigma(t, [f["sig_along"] for f in filas],
                       modo=modo_extrapolacion, modelo=modelo_sigma,
                       t_min=t_min_ajuste, verbose=verbose)
    if verbose:
        print("  ley de crecimiento cruzada:")
    ley_cr = ley_sigma(t, [f["sig_cross"] for f in filas],
                       modo=modo_extrapolacion, modelo=modelo_sigma,
                       t_min=t_min_ajuste, verbose=verbose)
    if sigma_z_ley is None:
        ley_z = dict(ley_cr, supuesto="sigma_z = sigma_cross (seccion isotropa)")
        if verbose:
            print("  [supuesto] sin Fase 2 no hay sigma_z medido: se usa el "
                  "sigma CRUZADO. Es el supuesto habitual de seccion isotropa "
                  "cerca de la fuente, y es un SUPUESTO, no una medicion.")
    else:
        ley_z = sigma_z_ley

    fuente = np.array([float(np.mean([f["centro"][0] for f in filas[:3]])),
                       float(np.mean([f["centro"][1] for f in filas[:3]])),
                       z_piso])
    if verbose:
        print(f"  fuente: E {fuente[0]:.0f}  N {fuente[1]:.0f}  "
              f"suelo {z_piso:.0f} m | altura efectiva H = {H:.1f} m")
        print(f"  viento: {u_ms:.2f} m/s HACIA {az_hacia:.1f} deg")
    return dict(fuente=fuente, H=H, u_ms=float(u_ms), pesado=pesos is not None,
                azimut_hacia=float(az_hacia), t=t, filas=filas, sig=sig,
                ley_along=ley_al, ley_cross=ley_cr, ley_z=ley_z,
                proyeccion=proj, calidad=calidad, ventana=ventana,
                z_piso=z_piso, t0_subframe=t0)


def _ang_dif(a: float, b: float) -> float:
    """a - b llevado a (-180, 180]."""
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _versores(azimut_hacia: float):
    """(e_along, e_cross) unitarios en el plano UTM. Shapes: (2,), (2,)."""
    a = np.radians(float(azimut_hacia))
    return (np.array([np.sin(a), np.cos(a)]),
            np.array([np.cos(a), -np.sin(a)]))


def perfil_3d_fase2(ruta_glb: str, azimut_hacia: float, ruta_serie: str = None,
                    fps: float = None, t0_frame: float = None,
                    verbose: bool = True) -> dict:
    """Momentos 3D del hull de la Fase 2, keyframe por keyframe."""
    import re
    import trimesh

    esc = trimesh.load(ruta_glb)
    geom = getattr(esc, "geometry", None)
    if not geom:
        raise ValueError(f"{os.path.basename(ruta_glb)} no trae geometrias por "
                         f"keyframe (se esperaba una escena con 'hull_fXXXXX').")

    t_csv, paso_vox = {}, None
    if ruta_serie and os.path.exists(ruta_serie):
        import csv as _csv
        pasos = []
        with open(ruta_serie, "r", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                t_csv[int(r["frame"])] = float(r["t_s"])
                try:
                    n = float(r["n_vox"])
                    if n > 0:
                        pasos.append((float(r["vol_m3"]) / n) ** (1 / 3))
                except (KeyError, ValueError):
                    pass
        if pasos:
            paso_vox = float(np.median(pasos))

    a = np.radians(float(azimut_hacia))
    e_al = np.array([np.sin(a), np.cos(a)])
    e_cr = np.array([np.cos(a), -np.sin(a)])
    T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)

    filas, sin_volumen = [], 0
    for nombre, m0 in geom.items():
        mm = re.search(r"(\d+)", nombre)
        if not mm:
            continue
        f = int(mm.group(1))
        m = m0.copy()
        if m.volume < 0:
            m.invert()
        if not m.is_volume:
            sin_volumen += 1
            continue
        V = float(m.volume)
        c = np.asarray(m.center_mass, float)
        I = np.asarray(m.moment_inertia, float)
        d = np.diag(I)
        M = np.array([[(d[1] + d[2] - d[0]) / 2, -I[0, 1], -I[0, 2]],
                      [-I[0, 1], (d[0] + d[2] - d[1]) / 2, -I[1, 2]],
                      [-I[0, 2], -I[1, 2], (d[0] + d[1] - d[2]) / 2]]) / V
        C = T @ M @ T.T
        Ch = C[:2, :2]
        w, v = np.linalg.eigh(Ch)
        may = v[:, int(np.argmax(w))]
        t = t_csv.get(f)
        if t is None and fps and t0_frame is not None:
            t = (f - float(t0_frame)) / float(fps)
        filas.append(dict(
            frame=f, t=t, vol_m3=V,
            cE=float(c[0]), cN=float(-c[2]), cU=float(c[1]),
            tope=float(np.asarray(m.vertices, float)[:, 1].max()),
            s_along=float(np.sqrt(max(e_al @ Ch @ e_al, 0.0))),
            s_cross=float(np.sqrt(max(e_cr @ Ch @ e_cr, 0.0))),
            s_z=float(np.sqrt(max(C[2, 2], 0.0))),
            az_mayor=float(np.degrees(np.arctan2(may[0], may[1])) % 180.0)))
    if not filas:
        raise ValueError("Ningun keyframe del GLB dio un volumen cerrado.")
    filas.sort(key=lambda r: r["frame"])
    if any(r["t"] is None for r in filas):
        raise ValueError("Sin tiempos: pasa `ruta_serie` (serie_columna.csv) o "
                         "`fps` y `t0_frame`.")

    per = {k: np.array([r[k] for r in filas], float)
           for k in ("frame", "t", "vol_m3", "cE", "cN", "cU", "tope",
                     "s_along", "s_cross", "s_z", "az_mayor")}
    per["n"] = len(filas)
    per["archivo"] = os.path.basename(ruta_glb)
    per["azimut_hacia"] = float(azimut_hacia)
    per["paso_voxel_m"] = paso_vox
    per["resuelto"] = (per["s_z"] >= 1.5 * paso_vox if paso_vox else
                       np.ones(per["n"], bool))
    if verbose:
        print(f"  [Fase 2] {per['n']} keyframes con volumen cerrado"
              + (f" ({sin_volumen} descartados)" if sin_volumen else "")
              + f" | t {per['t'].min():.1f} - {per['t'].max():.1f} s")
        print(f"           centroide sobre el piso: {per['cU'].min():.1f} - "
              f"{per['cU'].max():.1f} m | tope hasta {per['tope'].max():.1f} m")
        print(f"           sigma_z del hull: {per['s_z'].min():.1f} - "
              f"{per['s_z'].max():.1f} m")
        if paso_vox:
            print(f"           grilla de {paso_vox:.2f} m -> {int(per['resuelto'].sum())}"
                  f"/{per['n']} keyframes con forma resuelta "
                  f"(sigma_z >= 1,5 voxeles)")
    return per


def _en_ventana(per: dict, t_ini: float, t_fin: float):
    """Keyframes dentro de la ventana medida Y con forma resuelta."""
    m = (per["t"] >= t_ini) & (per["t"] <= t_fin) & per.get(
        "resuelto", np.ones(len(per["t"]), bool))
    if m.sum() >= 3:
        return m
    m = (per["t"] >= t_ini) & (per["t"] <= t_fin)
    return m if m.sum() >= 3 else np.ones(len(per["t"]), bool)


def resumen_fase2(per: dict, par: dict, ruta_camara: str = None,
                  verbose: bool = True) -> dict:
    """Las comprobaciones que deciden si el hull merece credito."""
    t_ini = float(par["ley_along"]["t_ini"])
    t_fin = float(par["ley_along"]["t_fin"])
    m = _en_ventana(per, t_ini, t_fin)
    t = per["t"][m]

    s_al_2d = np.asarray(par["ley_along"]["sigma"](t), float)
    s_cr_2d = np.asarray(par["ley_cross"]["sigma"](t), float)
    r_hull = per["s_along"][m] / np.maximum(per["s_cross"][m], 1e-6)
    r_2d = s_al_2d / np.maximum(s_cr_2d, 1e-6)
    k_ef = np.sqrt(5.0) * np.median(
        np.concatenate([per["s_along"][m] / s_al_2d, per["s_cross"][m] / s_cr_2d]))

    dt = t[-1] - t[0]
    dE = per["cE"][m][-1] - per["cE"][m][0]
    dN = per["cN"][m][-1] - per["cN"][m][0]
    vel_hull = float(np.hypot(dE, dN) / dt) if dt > 0 else float("nan")
    az_hull = float(np.degrees(np.arctan2(dE, dN)) % 360.0)

    sol = None
    if ruta_camara and os.path.exists(ruta_camara):
        try:
            with open(ruta_camara, "r", encoding="utf-8") as fh:
                d = json.load(fh).get("sol", {})
            if d:
                az_som = (float(d["azimut_deg"]) + 180.0) % 360.0
                sol = dict(azimut_sol_deg=float(d["azimut_deg"]),
                           elevacion_deg=float(d.get("elevacion_deg", float("nan"))),
                           azimut_sombra_deg=az_som,
                           angulo_sombra_viento_deg=abs(_ang_dif(az_som, par["azimut_hacia"])))
        except Exception as ex:
            print(f"  [aviso] no pude leer el sol de {os.path.basename(ruta_camara)}"
                  f" ({type(ex).__name__})")

    out = dict(
        ventana_s=[t_ini, t_fin], n=int(m.sum()), sol=sol,
        razon_horizontal_hull=float(np.median(r_hull)),
        razon_horizontal_2d=float(np.median(r_2d)),
        discrepancia_pct=float(100 * (np.median(r_hull) / np.median(r_2d) - 1)),
        corte_efectivo_sigmas=float(k_ef),
        razon_vertical_hull=float(np.median(per["s_z"][m] / np.maximum(per["s_cross"][m], 1e-6))),
        razon_vertical_rango=[float((per["s_z"][m] / per["s_cross"][m]).min()),
                              float((per["s_z"][m] / per["s_cross"][m]).max())],
        incertidumbre_razon_pct=float(abs(100 * (np.median(r_hull) / np.median(r_2d) - 1))),
        deriva_centroide=dict(vel_ms=vel_hull, azimut_hacia_deg=az_hull,
                              vel_fase3_ms=float(par["u_ms"]),
                              azimut_fase3_deg=float(par["azimut_hacia"]),
                              dif_dir_deg=_ang_dif(az_hull, par["azimut_hacia"]),
                              razon_vel=vel_hull / float(par["u_ms"])))
    if verbose:
        print("\n  COMPROBACIONES DEL HULL (antes de usarlo)")
        print(f"    razon horizontal along/cross : hull {out['razon_horizontal_hull']:.3f}"
              f"  vs  2D opacidad {out['razon_horizontal_2d']:.3f}"
              f"   ({out['discrepancia_pct']:+.1f} %)")
        if abs(out["discrepancia_pct"]) < 15:
            print("      -> concuerdan: el hull conserva la FORMA aunque su tamano "
                  "este recortado")
        elif abs(out["discrepancia_pct"]) < 40:
            print(f"      -> difieren {abs(out['discrepancia_pct']):.0f} %: es la "
                  f"INCERTIDUMBRE del metodo, no un veto.")
            print(f"         sigma_z queda determinado con esa banda "
                  f"(y chi del pico, que va como 1/sigma_z, tambien)")
        else:
            print(f"      -> difieren {abs(out['discrepancia_pct']):.0f} %: DEMASIADO. "
                  f"Revisar el carving antes de usar la razon vertical.")
        if out["sol"]:
            s_ = out["sol"]
            print(f"    sombra solar                 : cae hacia "
                  f"{s_['azimut_sombra_deg']:.0f} deg (sol a "
                  f"{s_['azimut_sol_deg']:.0f} deg, {s_['elevacion_deg']:.0f} deg "
                  f"de elevacion)")
            print(f"      -> {s_['angulo_sombra_viento_deg']:.0f} deg respecto del eje "
                  f"del viento" + ("  <- CASI ALINEADAS: el carving puede canjear "
                                   "altura por largo a favor del viento, y esa es la "
                                   "explicacion mas probable de la discrepancia de "
                                   "arriba" if s_["angulo_sombra_viento_deg"] < 30 else ""))
        print(f"    corte efectivo del carving   : ~{out['corte_efectivo_sigmas']:.1f} sigma "
              f"(el hull recorta mas apretado que la mascara 2D)")
        print(f"    razon vertical sigma_z/sigma_cross: {out['razon_vertical_hull']:.3f} "
              f"(rango {out['razon_vertical_rango'][0]:.2f}-{out['razon_vertical_rango'][1]:.2f})")
        print(f"      -> el supuesto anterior era 1,00 (seccion isotropa): "
              f"sobreestimaba sigma_z {1/max(out['razon_vertical_hull'],1e-6):.1f}x")
        d = out["deriva_centroide"]
        print(f"    deriva del centroide 3D      : {d['vel_ms']:.2f} m/s hacia "
              f"{d['azimut_hacia_deg']:.1f} deg")
        print(f"      -> Fase 3 (flujo optico)   : {d['vel_fase3_ms']:.2f} m/s hacia "
              f"{d['azimut_fase3_deg']:.1f} deg  ({d['dif_dir_deg']:+.1f} deg, "
              f"razon {d['razon_vel']:.2f})")
        print("      (estimador independiente; no entra en el modelo)")
    return out


def altura_efectiva_fase2(per: dict, t_fin_sigma: float = None,
                          frac_vol: float = 0.6, frac_meseta: float = 0.9,
                          verbose: bool = True) -> dict:
    """H = altura del centroide del hull donde deja de subir."""
    t, cU, vol = per["t"], per["cU"], per["vol_m3"]
    ok = vol >= float(frac_vol) * np.nanmax(vol)
    if ok.sum() < 3:
        ok = np.ones(len(t), bool)
    umbral = float(frac_meseta) * float(np.nanmax(cU[ok]))
    idx = np.flatnonzero(ok & (cU >= umbral))
    i0 = int(idx[0]) if len(idx) else int(np.argmax(cU))
    sel = ok & (np.arange(len(t)) >= i0)
    H = float(np.median(cU[sel]))
    out = dict(H_m=H, t_meseta_s=float(t[i0]), n_meseta=int(sel.sum()),
               H_rango=[float(cU[sel].min()), float(cU[sel].max())],
               t_valido_s=[float(t[ok][0]), float(t[ok][-1])])
    if t_fin_sigma is not None:
        m = (t >= 0) & (t <= t_fin_sigma)
        if m.sum() >= 2:
            out["subida_en_ventana_m"] = [float(cU[m][0]), float(cU[m][-1])]
            out["sigue_subiendo"] = bool(t[i0] > t_fin_sigma)
    if verbose:
        print(f"\n  ALTURA EFECTIVA H (centroide del hull)")
        print(f"    meseta desde t = {out['t_meseta_s']:.1f} s "
              f"({out['n_meseta']} keyframes) -> H = {H:.1f} m "
              f"(rango {out['H_rango'][0]:.1f}-{out['H_rango'][1]:.1f})")
        if "subida_en_ventana_m" in out:
            a_, b_ = out["subida_en_ventana_m"]
            print(f"    dentro de la ventana de sigma el centroide sube de "
                  f"{a_:.1f} a {b_:.1f} m")
            if out.get("sigue_subiendo"):
                print(f"    [supuesto] la columna AUN SUBE cuando termina la "
                      f"ventana de sigma: 'H constante' es del modelo, no del dato")
    return out


def corregir_plano_mapeo(par: dict, cam, per: dict, verbose: bool = True) -> tuple:
    """Rehace los momentos como si se hubieran mapeado al plano CORRECTO."""
    z_piso = float(par.get("z_piso", par["fuente"][2]))
    C = getattr(cam, "C", None)
    if C is None:
        C = getattr(cam, "C_utm", None)
    if C is None:
        raise AttributeError("La camara no expone su centro optico (cam.C).")
    C = np.asarray(C, float)
    h_cam = float(C[2]) - z_piso
    C_xy = C[:2]
    U_hull = np.interp([s["frame"] for s in par["sig"]], per["frame"], per["cU"])

    sig2, factores = [], []
    for s_, U in zip(par["sig"], U_hull):
        z_usado = float(s_["z"]) - z_piso
        f = (h_cam - float(U)) / max(h_cam - z_usado, 1e-6)
        c = np.asarray(s_["centro"], float)
        sig2.append(dict(s_, cov=np.asarray(s_["cov"], float) * f * f,
                         centro=tuple(C_xy + f * (c - C_xy)),
                         z=z_piso + float(U),
                         sig_may=s_["sig_may"] * f, sig_men=s_["sig_men"] * f))
        factores.append(f)
    factores = np.asarray(factores, float)
    info = dict(h_camara_m=h_cam, z_usado_m=float(np.mean(
        [float(s_["z"]) - z_piso for s_ in par["sig"]])),
        factor_ini=float(factores[0]), factor_fin=float(factores[-1]),
        factor_medio=float(np.mean(factores)))
    if verbose:
        print(f"\n  PLANO DE MAPEO (la Fase 3 lo fijo a "
              f"{info['z_usado_m']:.0f} m; la Fase 2 mide la altura real)")
        print(f"    camara a {h_cam:.0f} m sobre el piso -> homotecia de "
              f"{factores.min():.3f} a {factores.max():.3f} dentro de la ventana")
        print(f"    (f > 1 al principio porque la nube estaba MAS ABAJO del plano "
              f"supuesto,\n     f < 1 al final porque quedo mas arriba: no es una "
              f"escala, es la pendiente)")
    return sig2, info


def refinar_con_fase2(par: dict, cam, per: dict, iteraciones: int = 4,
                      usar_razon: bool = True, corregir_plano: bool = True,
                      verbose: bool = True) -> dict:
    """`par` con H y sigma_z medidos desde el hull y la proyeccion oblicua descontada.
    """
    t = np.asarray(par["t"], float)
    az = float(par["azimut_hacia"])

    sig_base, plano = par["sig"], None
    if corregir_plano:
        sig_base, plano = corregir_plano_mapeo(par, cam, per, verbose=verbose)

    res = per.get("resuelto", np.ones(per["n"], bool))
    if res.sum() < 3:
        res = np.ones(per["n"], bool)
    r_h = (per["s_z"] / np.maximum(per["s_cross"], 1e-6))[res]
    razon = np.interp(t, per["t"][res], r_h)
    sz_abs = np.interp(t, per["t"][res], per["s_z"][res])

    s_cr = np.array([f["sig_cross"] for f in par["filas"]], float)
    sz = razon * s_cr if usar_razon else sz_abs
    hist = []
    for _ in range(max(1, int(iteraciones))):
        filas, proj = descomponer_sigmas(sig_base, az, cam, sigma_z=sz,
                                         verbose=False)
        s_al = np.array([f["sig_along"] for f in filas], float)
        s_cr = np.array([f["sig_cross"] for f in filas], float)
        sz_new = razon * s_cr if usar_razon else sz_abs
        hist.append(float(np.max(np.abs(sz_new - sz))))
        if hist[-1] < 1e-4:
            sz = sz_new
            break
        sz = sz_new

    if verbose:
        print(f"\n  REFINAMIENTO (punto fijo sigma_z <-> sigma_cross): "
              f"{len(hist)} vueltas, ultimo cambio {hist[-1]:.2e} m")
        print("  leyes recalculadas con la degeneracion DESCONTADA:")
    _mod = par["ley_along"].get("modelo", "sigma0")
    _tmin = par["ley_along"].get("t_min")
    ley_al = ley_sigma(t, s_al, modo=par["ley_along"]["modo"], modelo=_mod,
                       t_min=_tmin, verbose=verbose)
    ley_cr = ley_sigma(t, s_cr, modo=par["ley_cross"]["modo"], modelo=_mod,
                       t_min=_tmin, verbose=verbose)
    ley_z = ley_sigma(t, sz, modo=par["ley_cross"]["modo"], modelo=_mod,
                      t_min=_tmin, verbose=verbose)
    ley_z["origen"] = ("razon sigma_z/sigma_cross del hull (Fase 2) x sigma_cross "
                       "medido (Fase 4)" if usar_razon else
                       "sigma_z absoluto del hull (Fase 2), sin corregir el corte")

    alt = altura_efectiva_fase2(per, t_fin_sigma=float(par["ley_along"]["t_fin"]),
                                verbose=verbose)

    _th = np.asarray(per["t"], float)
    _hh = np.asarray(per["cU"], float)
    _o = np.argsort(_th)
    _th, _hh = _th[_o], _hh[_o]

    def ley_H(tt, _t=_th, _h=_hh):
        return np.interp(np.asarray(tt, float), _t, _h)

    alt["via_modelo"] = "H(t) = centroide del hull, interpolado por instante"
    alt["H_t_rango_s"] = [float(_th.min()), float(_th.max())]
    alt["H_t_rango_m"] = [float(_hh.min()), float(_hh.max())]

    fuente = np.array([float(np.mean([f["centro"][0] for f in filas[:3]])),
                       float(np.mean([f["centro"][1] for f in filas[:3]])),
                       float(par["fuente"][2])])

    nuevo = dict(par)
    nuevo.update(filas=filas, proyeccion=proj, ley_along=ley_al,
                 ley_cross=ley_cr, ley_z=ley_z, H=float(alt["H_m"]),
                 ley_H=ley_H, sig=sig_base, fuente=fuente)
    ley_z_abs = ley_sigma(t, sz_abs, modo=par["ley_cross"]["modo"], verbose=False)

    nuevo["fase2"] = dict(
        archivo=per["archivo"], n_keyframes=int(per["n"]),
        n_resueltos=int(res.sum()), paso_voxel_m=per.get("paso_voxel_m"),
        sigma_z_fin_via_absoluta=float(ley_z_abs["sigma_fin"]),
        razon_vertical=float(np.median(razon)), via="razon" if usar_razon else "absoluto",
        iteraciones=len(hist), residuo_m=hist[-1], altura=alt,
        plano_mapeo=plano,
        desplazamiento_fuente_m=float(np.hypot(*(fuente[:2] - np.asarray(
            par["fuente"], float)[:2]))),
        antes=dict(H_m=float(par["H"]),
                   sigma_z="SUPUESTO = sigma_cross",
                   sigma_along_fin=float(par["ley_along"]["sigma_fin"]),
                   sigma_cross_fin=float(par["ley_cross"]["sigma_fin"]),
                   alpha_along=float(par["ley_along"]["alpha"]),
                   alpha_cross=float(par["ley_cross"]["alpha"]),
                   corregido=bool(par["proyeccion"]["corregido"])),
        despues=dict(H_m=float(alt["H_m"]),
                     sigma_z=f"MEDIDO (Fase 2), razon {float(np.median(razon)):.3f}",
                     sigma_along_fin=float(ley_al["sigma_fin"]),
                     sigma_cross_fin=float(ley_cr["sigma_fin"]),
                     sigma_z_fin=float(ley_z["sigma_fin"]),
                     alpha_along=float(ley_al["alpha"]),
                     alpha_cross=float(ley_cr["alpha"]),
                     alpha_z=float(ley_z["alpha"]), corregido=True))

    if verbose:
        print(f"    sigma_z por la via ABSOLUTA (hull sin corregir el corte): "
              f"{ley_z_abs['sigma_fin']:.1f} m")
        a_, d_ = nuevo["fase2"]["antes"], nuevo["fase2"]["despues"]
        print(f"\n  ANTES  ->  DESPUES   (al final de la ventana medida)")
        print(f"    H           : {a_['H_m']:6.1f} m  ->  {d_['H_m']:6.1f} m")
        print(f"    sigma_along : {a_['sigma_along_fin']:6.1f} m  ->  "
              f"{d_['sigma_along_fin']:6.1f} m")
        print(f"    sigma_cross : {a_['sigma_cross_fin']:6.1f} m  ->  "
              f"{d_['sigma_cross_fin']:6.1f} m")
        print(f"    sigma_z     : {a_['sigma_cross_fin']:6.1f} m  ->  "
              f"{d_['sigma_z_fin']:6.1f} m   (era el supuesto isotropo)")
        print(f"    alpha_along : {a_['alpha_along']:6.2f}    ->  "
              f"{d_['alpha_along']:6.2f}")
        print(f"    alpha_cross : {a_['alpha_cross']:6.2f}    ->  "
              f"{d_['alpha_cross']:6.2f}")
        if nuevo["fase2"]["desplazamiento_fuente_m"] > 0.5:
            print(f"    la fuente se corre {nuevo['fase2']['desplazamiento_fuente_m']:.1f} m "
                  f"en el suelo (la homotecia del plano)")
    return nuevo


def efecto_refinamiento(par_antes: dict, par_despues: dict,
                        tiempos=(5.0, 20.0, 60.0, 150.0),
                        z_receptor: float = 1.5, verbose: bool = True) -> dict:
    """Cambio de chi entre `par` sin y con el refinamiento de la Fase 2."""
    filas = []
    for t in np.atleast_1d(np.asarray(tiempos, float)):
        fila = dict(t_s=float(t))
        for etq, p in (("antes", par_antes), ("despues", par_despues)):
            f0 = np.asarray(p["fuente"], float)
            e_al, _ = _versores(p["azimut_hacia"])
            c = f0[:2] + float(p["u_ms"]) * float(t) * e_al
            P_res = np.array([[c[0], c[1], f0[2] + float(z_receptor)]])
            P_pic = np.array([[c[0], c[1], f0[2] + _altura_en(p, float(t))]])
            arg = (float(t), f0, float(p["u_ms"]), float(p["azimut_hacia"]),
                   p["ley_along"]["sigma"], p["ley_cross"]["sigma"],
                   p["ley_z"]["sigma"], _altura(p))
            fila[f"chi_receptor_{etq}"] = float(chi_puff(P_res, *arg)[0])
            fila[f"chi_pico_{etq}"] = float(chi_puff(P_pic, *arg)[0])
        for que in ("receptor", "pico"):
            a_ = fila[f"chi_{que}_antes"]
            fila[f"razon_{que}"] = (fila[f"chi_{que}_despues"] / a_
                                    if a_ > 0 else float("inf"))
        t_mes = par_despues.get("fase2", {}).get("altura", {}).get("t_meseta_s")
        fila["columna_aun_subia"] = bool(t_mes is not None and t < t_mes)
        filas.append(fila)
    if verbose:
        print(f"\n  EFECTO SOBRE chi  (z = {z_receptor:.1f} m y en el eje de la nube)")
        print(f"    {'t (s)':>6} {'chi 1,5m antes':>15} {'despues':>12} {'x':>7} "
              f"{'x en el pico':>14}")
        for f in filas:
            print(f"    {f['t_s']:6.0f} {f['chi_receptor_antes']:15.3e} "
                  f"{f['chi_receptor_despues']:12.3e} {f['razon_receptor']:7.2f} "
                  f"{f['razon_pico']:14.2f}"
                  + ("   <- la columna aun subia: H constante la pone mas alta "
                     "de lo que estaba" if f["columna_aun_subia"] else ""))
        print("    (>1 sube, <1 baja. Que el pico suba y lo de abajo baje NO es")
        print("     una contradiccion: la nube quedo mas compacta y mas alta.)")
    return dict(z_receptor=float(z_receptor), filas=filas)


def json_seguro(o):
    """`default=` para json.dump: numpy no es serializable y aqui hay arrays."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def serie_chi(par: dict, t_max: float = None, n: int = 400,
              z_receptor: float = 1.5) -> dict:
    """chi en el centro del puff: a la altura de la nube y a la de respiracion."""
    fuente = np.asarray(par["fuente"], float)
    u = float(par["u_ms"]); az = float(par["azimut_hacia"])
    H = _altura(par); t_fin = float(par["ley_along"]["t_fin"])
    e_al, _ = _versores(az)
    t_max = float(t_max if t_max else max(10 * t_fin, 150.0))

    def _chi(tt, z_rel):
        """z_rel = None sigue al centro de la nube, o sea z = H(t)."""
        tt = np.atleast_1d(np.asarray(tt, float))
        out = np.empty(len(tt))
        for i, t in enumerate(tt):
            c = fuente[:2] + u * t * e_al
            z_i = _altura_en(par, float(t)) if z_rel is None else z_rel
            P = np.array([[c[0], c[1], fuente[2] + z_i]])
            out[i] = chi_puff(P, float(t), fuente, u, az,
                              par["ley_along"]["sigma"], par["ley_cross"]["sigma"],
                              par["ley_z"]["sigma"], H)[0]
        return out

    t = np.geomspace(max(0.05, t_fin / 200), t_max, n)
    t_fr = np.asarray(par["t"], float)
    return dict(t=t, chi_nube=_chi(t, None), chi_receptor=_chi(t, z_receptor),
                t_frames=t_fr, chi_nube_frames=_chi(t_fr, None),
                chi_receptor_frames=_chi(t_fr, z_receptor),
                t_fin_medido=t_fin, z_receptor=float(z_receptor),
                H_m=_altura_en(par, t_fin), H_t=_altura(par))
