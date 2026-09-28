"""Fase 2: reconstruccion 3D de la columna por tallado espacial con dos vistas (camara y
sol), altura por sombra y exportacion a GLB y Blender.
"""

import datetime
import os
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.interpolate import RegularGridInterpolator

import funciones_F0 as F0


def latlon_to_dem_xy(lat: float, lon: float, dem_crs):
    """Lat/lon WGS84 -> coordenadas proyectadas del DEM."""
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def build_intrinsic_matrix(focal_px, width_px, height_px):
    cx = width_px / 2.0
    cy = height_px / 2.0
    return np.array([
        [focal_px, 0.0, cx],
        [0.0, focal_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

def make_H(R, t):
    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return H

def Rx(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)

def Ry(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)

def Rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

def build_extrinsic_intrinsic_matrix(frame_data, elevation, transform, crs):
    """Matrices extrinseca e intrinseca de la camara a partir de la telemetria y el DEM.
    """

    origin_x = transform.c
    origin_y = transform.f
    origin_z = elevation[0, 0]
    (origin_x, origin_y, origin_z)

    dron_x, dron_y = latlon_to_dem_xy(frame_data["lat_deg"], frame_data["lon_deg"], crs)
    (dron_x, dron_y, frame_data["absolute_altitude_m"])

    dem_origin_utm = np.array([origin_x, origin_y, origin_z], dtype=float)
    drone_point_utm = np.array([dron_x, dron_y, frame_data["absolute_altitude_m"]], dtype=float)

    K = build_intrinsic_matrix(frame_data["focal"], frame_data["width"], frame_data["height"])

    R_enu_from_ned_H = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)

    R_d_w = R_enu_from_ned_H @ (Rz(frame_data["drone_yaw_deg"]) @ Ry(0.0) @ Rx(0.0))
    t_d_w = drone_point_utm

    H_d_w = make_H(R_d_w, t_d_w)

    R_g_d = Rz(0.0) @ Ry(frame_data["gimbal_pitch_deg"]) @ Rx(0.0)
    t_g_d = np.array([0.0, -0.0803, 0.0136], dtype=float)

    H_g_d = make_H(R_g_d, t_g_d)

    R_c_g = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)

    t_c_g = np.array([0.0, -0.0421, 0.0])
    H_c_g = make_H(R_c_g, t_c_g)

    H_c_w = H_d_w @ H_g_d @ H_c_g
    return K, H_c_w, dem_origin_utm

def pixel_to_world_ray_from_H(u, v, K, H_c_w, world_origin_utm):
    """Rayo en mundo del pixel (u, v)."""
    pixel_h = np.array([u, v, 1.0], dtype=float)
    ray_c = np.linalg.solve(K, pixel_h)
    ray_c = ray_c / np.linalg.norm(ray_c)

    ray_w = H_c_w[:3, :3] @ ray_c
    ray_w = ray_w / np.linalg.norm(ray_w)
    ray_origin_utm = H_c_w[:3, 3]
    return ray_origin_utm, ray_w


def leer_dem_ventana(ruta_dem, bounds, verbose=True):
    """Lee la ventana `bounds` del DEM: arreglo, ejes e interpolador bilineal."""
    with rasterio.open(ruta_dem) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(f"DEM debe estar en CRS proyectado (UTM), no {src.crs}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError("DEM rotado: no soportado")

        ventana = from_bounds(*bounds, transform=src.transform)
        ventana = ventana.round_offsets().round_lengths()
        banda = src.read(1, window=ventana).astype(float)
        if src.nodata is not None:
            banda[banda == src.nodata] = np.nan
        tw = src.window_transform(ventana)
        crs = src.crs
        res_archivo = src.res
        shape_archivo = src.shape

    res_x, res_y = tw.a, -tw.e
    ny, nx = banda.shape

    xs = tw.c + res_x * (np.arange(nx) + 0.5)
    ys_desc = tw.f - res_y * (np.arange(ny) + 0.5)

    dem = np.flipud(banda)
    ys = ys_desc[::-1]

    Z_interp = RegularGridInterpolator(
        (ys, xs), dem, method="linear", bounds_error=False, fill_value=np.nan
    )

    if verbose:
        print(f"  archivo   : {shape_archivo} celdas, res {res_archivo} m, {crs}")
        print(f"  ventana   : {banda.shape} celdas ({banda.nbytes/1e6:.2f} MB)")
        print(f"  elevacion : {np.nanmin(dem):.0f} / {np.nanmax(dem):.0f} m  "
              f"(nan: {np.isnan(dem).mean()*100:.1f}%)")

    return dict(dem=dem, xs=xs, ys=ys, transform=tw, res=float(res_x),
                crs=crs, Z_interp=Z_interp, bounds=bounds)

def altura_dem(x, y, Z_interp):
    """Altura del terreno en (x, y) UTM. Acepta escalares o arrays."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    z = Z_interp(np.stack([y, x], axis=-1))
    return float(z[0]) if z.size == 1 else z


def interseccion_rayos_dem(origenes, direcciones, Z_interp,
                           t_min=0.0, t_max=4000.0, paso=5.0, n_bisec=25):
    """Interseccion de N rayos con el DEM por ray-marching vectorizado + biseccion."""
    O = np.atleast_2d(np.asarray(origenes, float))
    D = np.atleast_2d(np.asarray(direcciones, float))
    if len(O) == 1 and len(D) > 1:
        O = np.repeat(O, len(D), axis=0)
    if len(D) == 1 and len(O) > 1:
        D = np.repeat(D, len(O), axis=0)
    n = len(O)

    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    def f_sub(t, idx):
        """terreno - rayo, evaluado solo en el subconjunto idx."""
        P = O[idx] + t[:, None] * D[idx]
        z = Z_interp(np.stack([P[:, 1], P[:, 0]], axis=-1))
        return z - P[:, 2]

    a = np.full(n, np.nan)
    b = np.full(n, np.nan)

    pend = np.arange(n)
    t_prev = np.full(n, t_min)
    v_prev = f_sub(np.full(n, t_min), pend)

    t = t_min + paso
    while t <= t_max and pend.size:
        v_cur = f_sub(np.full(pend.size, t), pend)

        cruce = (np.isfinite(v_prev) & np.isfinite(v_cur) &
                 (v_prev * v_cur <= 0.0))
        if cruce.any():
            gi = pend[cruce]
            a[gi] = t_prev[gi]
            b[gi] = t

        sigue = ~cruce
        pend = pend[sigue]
        v_prev = v_cur[sigue]
        t_prev[pend] = t
        t += paso

    ok = np.isfinite(a)
    idx = np.flatnonzero(ok)
    t_hit = np.full(n, np.nan)

    if idx.size:
        aa, bb = a[idx].copy(), b[idx].copy()
        s_a = np.sign(f_sub(aa, idx))
        s_a[s_a == 0] = 1.0
        for _ in range(n_bisec):
            m = 0.5 * (aa + bb)
            v_m = f_sub(m, idx)
            cond = (s_a * v_m <= 0.0)
            bb = np.where(cond, m, bb)
            aa = np.where(cond, aa, m)
        t_hit[idx] = 0.5 * (aa + bb)

    puntos = O + t_hit[:, None] * D
    puntos[~np.isfinite(t_hit)] = np.nan
    return puntos, t_hit


def construir_camara(frame_data, escala=1.0, dem_ctx=None, verbose=True):
    """K y H_c_w de la camara, con la intrinseca escalada a la resolucion de trabajo."""
    fd = dict(frame_data)
    fd["width"]  = int(round(frame_data["width"]  * escala))
    fd["height"] = int(round(frame_data["height"] * escala))
    fd["focal"]  = frame_data["focal"] * escala

    elev_dummy = np.zeros((2, 2))
    if dem_ctx is not None:
        transform, crs = dem_ctx["transform"], dem_ctx["crs"]
    else:
        raise ValueError("dem_ctx es obligatorio (aporta transform y crs)")

    K, H_c_w, dem_origin_utm = build_extrinsic_intrinsic_matrix(
        fd, elev_dummy, transform, crs
    )

    if verbose:
        pos = H_c_w[:3, 3]
        d_c = mirada_camara(K, H_c_w, fd["width"] / 2.0, fd["height"] / 2.0)
        az = np.degrees(np.arctan2(d_c[0], d_c[1])) % 360.0
        dep = np.degrees(np.arcsin(np.clip(-d_c[2], -1, 1)))
        fov_h = 2 * np.degrees(np.arctan(fd["width"]  / (2 * fd["focal"])))
        fov_v = 2 * np.degrees(np.arctan(fd["height"] / (2 * fd["focal"])))
        print(f"  resolucion : {fd['width']}x{fd['height']}  f={fd['focal']:.1f} px")
        print(f"  FOV        : {fov_h:.1f} x {fov_v:.1f} deg")
        print(f"  dron UTM   : E {pos[0]:.1f}  N {pos[1]:.1f}  z {pos[2]:.1f} m")
        print(f"  mirada     : azimut {az:.1f} deg | depresion {dep:.1f} deg")

    return K, H_c_w, fd

def mirada_camara(K, H_c_w, u, v):
    """Direccion unitaria en mundo del rayo que pasa por el pixel (u, v)."""
    _, d = pixel_to_world_ray_from_H(u, v, K, H_c_w, None)
    return d

def pixel_a_rayo(us, vs, K, H_c_w):
    """Rayos en mundo de los pixeles (us, vs). Devuelve (origen (3,), direcciones (N,3)).
    """
    us = np.atleast_1d(np.asarray(us, float)).ravel()
    vs = np.atleast_1d(np.asarray(vs, float)).ravel()
    pix = np.stack([us, vs, np.ones_like(us)], axis=1)
    rc = pix @ np.linalg.inv(K).T
    rc /= np.linalg.norm(rc, axis=1, keepdims=True)
    rw = rc @ H_c_w[:3, :3].T
    rw /= np.linalg.norm(rw, axis=1, keepdims=True)
    return H_c_w[:3, 3].copy(), rw

def mundo_a_pixel(P_utm, K, H_c_w):
    """Proyeccion mundo -> pixel."""
    P = np.atleast_2d(np.asarray(P_utm, float))
    R = H_c_w[:3, :3]
    t = H_c_w[:3, 3]
    P_cam = (P - t) @ R
    delante = P_cam[:, 2] > 1e-9
    z = np.where(delante, P_cam[:, 2], np.nan)
    uv = np.stack([
        K[0, 0] * P_cam[:, 0] / z + K[0, 2],
        K[1, 1] * P_cam[:, 1] / z + K[1, 2],
    ], axis=1)
    return uv, delante


def posicion_solar(dt_local, lat_deg, lon_deg, verbose=True):
    """Posicion del sol via pysolar."""
    from pysolar.solar import get_altitude, get_azimuth

    el = get_altitude(lat_deg, lon_deg, dt_local)
    az = get_azimuth(lat_deg, lon_deg, dt_local)

    if el <= 0:
        raise ValueError(f"Sol bajo el horizonte (elevacion {el:.2f} deg): "
                         "revisa la fecha/hora o la zona horaria.")

    a, e = np.radians(az), np.radians(el)
    vec = np.array([np.cos(e) * np.sin(a),
                    np.cos(e) * np.cos(a),
                    np.sin(e)])
    vec /= np.linalg.norm(vec)

    if verbose:
        sombra = -vec.copy(); sombra[2] = 0.0
        sombra /= np.linalg.norm(sombra)
        az_sombra = np.degrees(np.arctan2(sombra[0], sombra[1])) % 360.0
        _off = dt_local.utcoffset()
        _h = _off.total_seconds() / 3600.0 if _off is not None else 0.0
        print(f"  instante   : {dt_local:%Y-%m-%d %H:%M:%S}  (UTC{_h:+.0f})")
        print(f"  sol        : azimut {az:.2f} deg | elevacion {el:.2f} deg")
        print(f"  vec_sol    : E {vec[0]:+.4f}  N {vec[1]:+.4f}  U {vec[2]:+.4f}")
        print(f"  sombra hacia azimut {az_sombra:.1f} deg | "
              f"largo = {1/np.tan(e):.2f} x altura")

    return dict(azimut_deg=float(az), elevacion_deg=float(el), vec_sol=vec)


def mapas_terreno_xyz(K, H_c_w, width, height, Z_interp,
                  paso=2.0, t_max=3000.0, n_bisec=25, step=1,
                  chunk=400_000, verbose=True):
    """Traza un rayo por pixel y devuelve los mapas UTM del terreno visto."""
    import time

    us = np.arange(0, width, step)
    vs = np.arange(0, height, step)
    UU, VV = np.meshgrid(us, vs)
    forma = UU.shape
    n = UU.size

    if verbose:
        print(f"  trazando {n:,} rayos ({forma[1]}x{forma[0]}, step={step}, "
              f"paso={paso} m)...")

    salida = np.full((n, 3), np.nan)
    t0 = time.time()
    for ini in range(0, n, chunk):
        fin = min(n, ini + chunk)
        O, D = pixel_a_rayo(UU.ravel()[ini:fin], VV.ravel()[ini:fin], K, H_c_w)
        P, _ = interseccion_rayos_dem(O, D, Z_interp, t_max=t_max,
                                      paso=paso, n_bisec=n_bisec)
        salida[ini:fin] = P
        if verbose:
            print(f"    {fin:>9,}/{n:,}  ({time.time()-t0:6.1f} s)", end="\r")

    x_map = salida[:, 0].reshape(forma)
    y_map = salida[:, 1].reshape(forma)
    z_map = salida[:, 2].reshape(forma)

    if verbose:
        ok = np.isfinite(x_map)
        print(f"\n  aciertos : {ok.mean()*100:.1f}%  ({time.time()-t0:.1f} s)")
        if ok.any():
            print(f"  huella   : E [{np.nanmin(x_map):.0f}, {np.nanmax(x_map):.0f}]"
                  f"  N [{np.nanmin(y_map):.0f}, {np.nanmax(y_map):.0f}]"
                  f"  ({np.nanmax(x_map)-np.nanmin(x_map):.0f} x "
                  f"{np.nanmax(y_map)-np.nanmin(y_map):.0f} m)")
            print(f"  z terreno: {np.nanmin(z_map):.0f} / {np.nanmax(z_map):.0f} m")

    return x_map, y_map, z_map

def sombra_estatica(masks_sombra, frames_pre, umbral=0.5, verbose=True):
    """Sombra permanente del terreno: fraccion de frames pre-tronadura en que cada pixel es
    sombra.
    """
    frames_pre = [f for f in frames_pre if f in masks_sombra]
    if not frames_pre:
        raise ValueError(
            "No hay frames PRE-tronadura en las mascaras de sombra. "
            "Re-ejecuta extract_shadow_masks_multichannel en V6 con "
            "start_frame = PIVOT_FRAME - 40.")

    acc = np.zeros(masks_sombra[frames_pre[0]].shape, dtype=np.float32)
    for f in frames_pre:
        acc += masks_sombra[f]
    frac = acc / len(frames_pre)
    estatica = frac >= umbral

    if verbose:
        print(f"  frames pre-tronadura : {len(frames_pre)} "
              f"[f{min(frames_pre)} .. f{max(frames_pre)}]")
        print(f"  sombra estatica      : {estatica.mean()*100:.2f}% del frame "
              f"(umbral fraccion >= {umbral})")
    return estatica, frac


def crear_grilla_voxeles(centro, radio_xy, z_rel_min, z_rel_max, res, Z_interp,
                         verbose=True):
    """Grilla de voxeles centrada en `centro`; cotas relativas al terreno bajo el centro,
    sin voxeles bajo el terreno.
    """
    cx, cy = float(centro[0]), float(centro[1])
    z0 = altura_dem(cx, cy, Z_interp)

    vx = np.arange(cx - radio_xy, cx + radio_xy + res, res)
    vy = np.arange(cy - radio_xy, cy + radio_xy + res, res)
    vz = np.arange(z0 + z_rel_min, z0 + z_rel_max + res, res)
    nx, ny, nz = len(vx), len(vy), len(vz)

    GX, GY = np.meshgrid(vx, vy, indexing="ij")
    z_terr = altura_dem(GX.ravel(), GY.ravel(), Z_interp).reshape(nx, ny)

    I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                           indexing="ij")
    P = np.stack([vx[I].ravel(), vy[J].ravel(), vz[Kk].ravel()], axis=1)
    idx = np.stack([I.ravel(), J.ravel(), Kk.ravel()], axis=1)

    sobre = P[:, 2] > np.repeat(z_terr.ravel(), nz)
    P, idx = P[sobre], idx[sobre]

    if verbose:
        print(f"  centro   : E {cx:.1f}  N {cy:.1f}  z terreno {z0:.1f} m")
        print(f"  grilla   : {nx} x {ny} x {nz} = {nx*ny*nz:,} celdas, res {res} m")
        print(f"  sobre terreno: {len(P):,} voxeles ({len(P)/(nx*ny*nz)*100:.1f}%)")
        print(f"  extension: +-{radio_xy} m en XY | z rel [{z_rel_min}, {z_rel_max}] m")

    return dict(P=P, idx=idx, vx=vx, vy=vy, vz=vz, forma=(nx, ny, nz),
                res=float(res), z_terr=z_terr, z0=float(z0))


def hull_a_malla(G, ocupado, origen, suavizar=1):
    """Marching cubes sobre la ocupacion. Devuelve (vertices, caras) en coordenadas locales
    relativas a `origen`.
    """
    from skimage import measure
    from scipy.ndimage import gaussian_filter

    nx, ny, nz = G["forma"]
    vol = np.zeros((nx, ny, nz), np.float32)
    idx = G["idx"][ocupado]
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    if vol.sum() < 8:
        return None, None
    vol = np.pad(vol, 1)
    if suavizar:
        vol = gaussian_filter(vol, sigma=suavizar)

    try:
        v, f, _, _ = measure.marching_cubes(vol, level=0.5)
    except (ValueError, RuntimeError):
        return None, None

    v = v - 1.0
    V = np.stack([G["vx"][0] + v[:, 0] * G["res"],
                  G["vy"][0] + v[:, 1] * G["res"],
                  G["vz"][0] + v[:, 2] * G["res"]], axis=1)
    return V - np.asarray(origen, float)[None, :], f

def exportar_hulls_glb(hulls, G, origen, ruta_salida, suavizar=1,
                       color=(120, 120, 130, 90), verbose=True):
    """Un objeto por keyframe en una escena GLB. Nombres 'hull_fXXXXX'."""
    import trimesh
    escena = trimesh.Scene()
    resumen = []
    for f in sorted(hulls):
        V, F = hull_a_malla(G, hulls[f], origen, suavizar=suavizar)
        if V is None:
            resumen.append((f, 0, 0.0)); continue
        Vg = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)
        m = trimesh.Trimesh(vertices=Vg, faces=F, process=False)
        m.visual.vertex_colors = np.tile(np.array(color, np.uint8), (len(Vg), 1))
        escena.add_geometry(m, node_name=f"hull_f{f:05d}",
                            geom_name=f"hull_f{f:05d}")
        resumen.append((f, len(F), float(m.volume) if m.is_volume else np.nan))

    escena.export(ruta_salida)
    if verbose:
        tot = sum(r[1] for r in resumen)
        vac = sum(1 for r in resumen if r[1] == 0)
        import os
        print(f"  {len(resumen)} keyframes | {tot:,} triangulos | "
              f"{vac} vacios | {os.path.getsize(ruta_salida)/1e6:.1f} MB")
        print(f"  origen UTM restado: E {origen[0]:.2f} N {origen[1]:.2f} "
              f"z {origen[2]:.2f}")
        print(f"  ejes en Blender: +X = Este, +Y = Norte, +Z = arriba")
        print(f"  escrito: {F0.ruta_corta(ruta_salida)}")
    return resumen

def exportar_camara_json(K, H_c_w, origen, W, H, ruta_salida, sol=None,
                         fps=None, keyframes=None, color_gas=None,
                         script_blender=True):
    """Pose de camara y sol, en coordenadas locales, para el script de Blender."""
    import json
    pos = H_c_w[:3, 3] - np.asarray(origen, float)
    R = H_c_w[:3, :3]
    d = {
        "posicion_blender": [float(pos[0]), float(pos[1]), float(pos[2])],
        "R_c_w": R.tolist(),
        "focal_px": float(K[0, 0]), "ancho_px": int(W), "alto_px": int(H),
        "fov_h_deg": float(2*np.degrees(np.arctan(W/(2*K[0, 0])))),
        "sensor_mm": 36.0,
        "focal_mm": float(36.0 * K[0, 0] / W),
        "clip_end": 20000.0,
        "origen_utm": [float(c) for c in origen],
    }
    if sol is not None:
        d["sol"] = {"azimut_deg": float(sol["azimut_deg"]),
                    "elevacion_deg": float(sol["elevacion_deg"]),
                    "vec_sol": [float(c) for c in sol["vec_sol"]]}
    if fps: d["fps"] = float(fps)
    if keyframes is not None: d["keyframes"] = [int(k) for k in keyframes]
    if color_gas:
        d["color_gas_srgb"] = {int(k): [float(c) for c in v]
                               for k, v in color_gas.items()}
    with open(ruta_salida, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
    if script_blender:
        escribir_script_blender(os.path.dirname(os.path.abspath(ruta_salida)))
    return d

def exportar_terreno_glb(DEM, origen, ruta_salida, radio=800.0,
                         ruta_textura=None, paso=1, verbose=True):
    """Malla del terreno alrededor de `origen`, con la misma convencion de ejes que
    exportar_hulls_glb.
    """
    import trimesh
    xs, ys, dem = DEM["xs"], DEM["ys"], DEM["dem"]

    ix = np.flatnonzero((xs >= origen[0] - radio) & (xs <= origen[0] + radio))[::paso]
    iy = np.flatnonzero((ys >= origen[1] - radio) & (ys <= origen[1] + radio))[::paso]
    if len(ix) < 2 or len(iy) < 2:
        raise ValueError("El radio no cubre celdas del DEM; revisa origen/bounds.")

    X, Y = np.meshgrid(xs[ix], ys[iy], indexing="ij")
    Z = dem[np.ix_(iy, ix)].T
    nx, ny = X.shape

    V = np.stack([X.ravel() - origen[0], Y.ravel() - origen[1],
                  np.nan_to_num(Z, nan=float(np.nanmin(Z))).ravel() - origen[2]],
                 axis=1)

    I, J = np.meshgrid(np.arange(nx-1), np.arange(ny-1), indexing="ij")
    a = (I*ny + J).ravel(); b = a + ny; c = b + 1; d = a + 1
    F = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    Vg = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)
    malla = trimesh.Trimesh(vertices=Vg, faces=F, process=False)

    if ruta_textura is not None:
        try:
            from PIL import Image
            with rasterio.open(ruta_textura) as src:
                img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
                bx0, by0, bx1, by1 = src.bounds.left, src.bounds.bottom, \
                                     src.bounds.right, src.bounds.top
            u = (X.ravel() - bx0) / max(bx1 - bx0, 1e-9)
            v = (Y.ravel() - by0) / max(by1 - by0, 1e-9)
            malla.visual = trimesh.visual.TextureVisuals(
                uv=np.clip(np.stack([u, v], 1), 0, 1),
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(img.astype(np.uint8)),
                    metallicFactor=0.0, roughnessFactor=1.0))
            if verbose:
                print(f"  textura   : {os.path.basename(ruta_textura)} "
                      f"({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            print(f"  aviso: no se pudo aplicar la textura ({e}); terreno gris.")

    escena = trimesh.Scene()
    escena.add_geometry(malla, node_name="terreno", geom_name="terreno")
    escena.export(ruta_salida)

    if verbose:
        import os as _os
        print(f"  malla     : {nx}x{ny} vertices, {len(F):,} triangulos")
        print(f"  extension : +-{radio:.0f} m | z local "
              f"[{V[:,2].min():.1f}, {V[:,2].max():.1f}] m")
        print(f"  archivo   : {_os.path.getsize(ruta_salida)/1e6:.1f} MB -> {F0.ruta_corta(ruta_salida)}")
    return dict(n_vert=int(len(V)), n_caras=int(len(F)),
                z_min=float(V[:, 2].min()), z_max=float(V[:, 2].max()))

def sombra_terreno(P, vec_sol, Z_interp, t_max=1500.0, paso=10.0):
    """Auto-sombra del terreno: True si el rayo al sol queda bajo la superficie."""
    P = np.atleast_2d(np.asarray(P, float))
    d = np.asarray(vec_sol, float)
    oculto = np.zeros(len(P), bool)
    for i in range(1, int(t_max / paso) + 1):
        Q = P + (i * paso) * d[None, :]
        zt = Z_interp(np.stack([Q[:, 1], Q[:, 0]], axis=-1))
        oculto |= np.isfinite(zt) & (zt > Q[:, 2])
    return oculto


def denso(G, ocupado):
    """Ocupacion dispersa (N,) -> arreglo booleano (nx, ny, nz)."""
    vol = np.zeros(G["forma"], bool)
    ij = G["idx"][ocupado]
    vol[ij[:, 0], ij[:, 1], ij[:, 2]] = True
    return vol

def disperso(G, vol):
    """Arreglo (nx, ny, nz) -> ocupacion dispersa (N,) sobre G['P']."""
    ij = G["idx"]
    return vol[ij[:, 0], ij[:, 1], ij[:, 2]]

def muestrear(P, G, vol):
    """Ocupacion en puntos UTM arbitrarios (vecino mas cercano)."""
    nx, ny, nz = vol.shape
    ix = np.rint((P[:, 0] - G["vx"][0]) / G["res"]).astype(np.int32)
    iy = np.rint((P[:, 1] - G["vy"][0]) / G["res"]).astype(np.int32)
    iz = np.rint((P[:, 2] - G["vz"][0]) / G["res"]).astype(np.int32)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) &
          (iz >= 0) & (iz < nz))
    out = np.zeros(len(P), bool)
    if ok.any():
        out[ok] = vol[ix[ok], iy[ok], iz[ok]]
    return out

def altura_sobre_terreno(G, ocupado=None):
    """Altura de cada voxel sobre el terreno de su propia columna xy."""
    sel = slice(None) if ocupado is None else ocupado
    P = G["P"][sel]
    ij = G["idx"][sel]
    return P[:, 2] - G["z_terr"][ij[:, 0], ij[:, 1]]


def profundidad_terreno(XM, YM, ZM, H_c_w):
    """Profundidad de camara del terreno en cada pixel (inf donde el rayo no lo corta).
    """
    R, t = H_c_w[:3, :3], H_c_w[:3, 3]
    fin = np.isfinite(XM) & np.isfinite(YM) & np.isfinite(ZM)
    prof = np.full(XM.shape, np.inf)
    if fin.any():
        P = np.stack([XM[fin], YM[fin], ZM[fin]], axis=1)
        prof[fin] = ((P - t) @ R)[:, 2]
    return prof, fin

def mapa_autosombra(XM, YM, ZM, vec_sol, Z_interp, t_max=1500.0, paso=8.0,
                    verbose=True):
    """Pixeles de terreno que el propio relieve deja sin sol."""
    t0 = time.time()
    fin = np.isfinite(XM) & np.isfinite(YM) & np.isfinite(ZM)
    out = np.zeros(XM.shape, bool)
    if fin.any():
        P = np.stack([XM[fin], YM[fin], ZM[fin]], axis=1)
        out[fin] = sombra_terreno(P, vec_sol, Z_interp,
                                     t_max=t_max, paso=paso)
    if verbose:
        print(f"  auto-sombra del terreno : {out.mean()*100:5.2f}% del cuadro "
              f"({time.time()-t0:.1f} s)")
    return out

def evidencia_sombra(m_som_col, m_gas, m_estatica, m_autosombra, sin_terreno,
                     radio_estatica=5, radio_gas=3):
    """Mapa ternario por pixel: +1 sombra de la columna, 0 suelo iluminado observable, -1
    desconocido.
    """
    import cv2
    def _k(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)

    desc = (cv2.dilate(m_estatica.astype(np.uint8), _k(radio_estatica)).astype(bool)
            | cv2.dilate(m_gas.astype(np.uint8), _k(radio_gas)).astype(bool)
            | m_autosombra | sin_terreno)

    ev = np.zeros(m_som_col.shape, np.int8)
    ev[desc] = -1
    ev[m_som_col & ~desc] = 1
    return ev


def precalcular(G, K, H_c_w, vec_sol, Z_interp, XM, YM, ZM, W, H,
                paso=None, t_max=800.0, margen_prof=None, verbose=True):
    """Camara y sol estaticos => esto se calcula UNA vez para todos los frames."""
    P = G["P"]
    n = len(P)
    if paso is None:
        paso = G["res"]
    if margen_prof is None:
        margen_prof = 1.5 * G["res"]
    t0 = time.time()

    prof_ter, hay_ter = profundidad_terreno(XM, YM, ZM, H_c_w)
    R, t = H_c_w[:3, :3], H_c_w[:3, 3]

    uv_c, delante = mundo_a_pixel(P, K, H_c_w)
    uv_c = np.nan_to_num(uv_c, nan=-1e9)
    uc = np.round(uv_c[:, 0]).astype(np.int64)
    vc = np.round(uv_c[:, 1]).astype(np.int64)
    en_cuadro = delante & (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)

    ucc = np.clip(uc, 0, W - 1)
    vcc = np.clip(vc, 0, H - 1)
    prof_vox = ((P - t) @ R)[:, 2]
    lim = np.where(en_cuadro, prof_ter[vcc, ucc], np.inf)
    ok_c = en_cuadro & (prof_vox < lim + margen_prof)

    if verbose:
        print(f"  camara  : {en_cuadro.mean()*100:5.1f}% en cuadro -> "
              f"{ok_c.mean()*100:5.1f}% tras oclusion")

    D = np.repeat((-np.asarray(vec_sol, float))[None, :], n, axis=0)
    Pg, tg = interseccion_rayos_dem(P, D, Z_interp, t_max=t_max,
                                       paso=paso, n_bisec=20)

    uv_s, delante_s = mundo_a_pixel(np.nan_to_num(Pg, nan=0.0), K, H_c_w)
    uv_s = np.nan_to_num(uv_s, nan=-1e9)
    us = np.round(uv_s[:, 0]).astype(np.int64)
    vs = np.round(uv_s[:, 1]).astype(np.int64)
    en_cuadro_s = (np.isfinite(tg) & delante_s &
                   (us >= 0) & (us < W) & (vs >= 0) & (vs < H))

    usc = np.clip(us, 0, W - 1)
    vsc = np.clip(vs, 0, H - 1)
    prof_g = np.full(n, np.inf)
    if en_cuadro_s.any():
        prof_g[en_cuadro_s] = ((Pg[en_cuadro_s] - t) @ R)[:, 2]
    lim_s = np.where(en_cuadro_s, prof_ter[vsc, usc], -np.inf)
    ok_s = en_cuadro_s & (prof_g < lim_s + margen_prof)

    if verbose:
        print(f"  sol     : {en_cuadro_s.mean()*100:5.1f}% aterriza en cuadro -> "
              f"{ok_s.mean()*100:5.1f}% con suelo visible")
        print(f"  ({time.time()-t0:.1f} s)")

    return dict(uc=ucc, vc=vcc, ok_c=ok_c,
                us=usc, vs=vsc, ok_s=ok_s, P_suelo=Pg,
                prof_terreno=prof_ter, hay_terreno=hay_ter)


def carving(G, PR, mask_gas, ev_som, usar_sombra=True):
    """Space carving de dos vistas con evidencia ternaria."""
    n = len(G["P"])
    en_gas = np.zeros(n, bool)
    en_gas[PR["ok_c"]] = mask_gas[PR["vc"][PR["ok_c"]], PR["uc"][PR["ok_c"]]]

    libre = np.zeros(n, bool)
    if usar_sombra:
        o = PR["ok_s"]
        libre[o] = (ev_som[PR["vs"][o], PR["us"][o]] == 0)

    ocupado = en_gas & ~libre
    diag = dict(n_gas=int(en_gas.sum()), n_libre=int(libre.sum()),
                uso_sombra=bool(usar_sombra),
                n_ocupado=int(ocupado.sum()),
                frac_sin_info=float((~PR["ok_s"] & en_gas).sum()
                                    / max(int(en_gas.sum()), 1)))
    return ocupado, diag

def regularizar(G, ocupado, cerrar=1, abrir=1, solo_mayor=True,
                min_abrir=400):
    """Limpia el resultado del carving."""
    from scipy import ndimage
    vol = denso(G, ocupado)
    if not vol.any():
        return ocupado.copy(), dict(n_ini=0, n_fin=0, n_comp=0)
    n_ini = int(vol.sum())
    if cerrar:
        vol = ndimage.binary_closing(vol, iterations=int(cerrar))
    aplico_abrir = bool(abrir) and n_ini >= int(min_abrir)
    if aplico_abrir:
        vol = ndimage.binary_opening(vol, iterations=int(abrir))
        if not vol.any():
            vol = denso(G, ocupado)
            if cerrar:
                vol = ndimage.binary_closing(vol, iterations=int(cerrar))
            aplico_abrir = False
    n_comp = 0
    if solo_mayor and vol.any():
        lab, n_comp = ndimage.label(vol)
        if n_comp > 1:
            tam = ndimage.sum(vol, lab, index=np.arange(1, n_comp + 1))
            vol = (lab == (int(np.argmax(tam)) + 1))
    salida = disperso(G, vol)
    return salida, dict(n_ini=n_ini, n_fin=int(salida.sum()),
                        n_comp=int(n_comp), abrio=aplico_abrir)


def silueta(G, ocupado, K, H_c_w, W, H):
    """Silueta del hull en la imagen, proyectando las 8 esquinas de cada voxel."""
    P = G["P"][ocupado]
    if len(P) == 0:
        return np.zeros((H, W), bool)

    r = G["res"] / 2.0
    esq = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)], float) * r

    U = np.empty((len(P), 8)); V = np.empty((len(P), 8))
    Dl = np.empty((len(P), 8), bool)
    for e in range(8):
        uv, d = mundo_a_pixel(P + esq[e], K, H_c_w)
        U[:, e] = np.nan_to_num(uv[:, 0], nan=-1e9)
        V[:, e] = np.nan_to_num(uv[:, 1], nan=-1e9)
        Dl[:, e] = d
    val = Dl.all(axis=1)
    if not val.any():
        return np.zeros((H, W), bool)

    u0 = np.clip(np.floor(U[val].min(1)), 0, W - 1).astype(np.int64)
    u1 = np.clip(np.ceil(U[val].max(1)), 0, W - 1).astype(np.int64)
    v0 = np.clip(np.floor(V[val].min(1)), 0, H - 1).astype(np.int64)
    v1 = np.clip(np.ceil(V[val].max(1)), 0, H - 1).astype(np.int64)

    dif = np.zeros((H + 1, W + 1), np.int32)
    np.add.at(dif, (v0, u0), 1)
    np.add.at(dif, (v0, u1 + 1), -1)
    np.add.at(dif, (v1 + 1, u0), -1)
    np.add.at(dif, (v1 + 1, u1 + 1), 1)
    return np.cumsum(np.cumsum(dif, axis=0), axis=1)[:H, :W] > 0


def sombra_proyectada(G, ocupado, XM, YM, ZM, vec_sol, step=2, paso=None,
                      t_max=700.0):
    """Sombra que el hull proyecta sobre el terreno, vista desde la camara."""
    import cv2
    if paso is None:
        paso = G["res"] * 0.6
    H, W = XM.shape
    vol = denso(G, ocupado)
    if not vol.any():
        return np.zeros((H, W), bool)

    Xs, Ys, Zs = XM[::step, ::step], YM[::step, ::step], ZM[::step, ::step]
    hs, ws = Xs.shape
    val = np.isfinite(Xs) & np.isfinite(Ys) & np.isfinite(Zs)
    P = np.stack([np.nan_to_num(Xs).ravel(), np.nan_to_num(Ys).ravel(),
                  np.nan_to_num(Zs).ravel()], axis=1)

    idx = np.flatnonzero(val.ravel())
    som = np.zeros(hs * ws, bool)
    pend = np.ones(len(idx), bool)
    d = np.asarray(vec_sol, float)
    for i in range(1, int(t_max / paso) + 1):
        if not pend.any():
            break
        sub = np.flatnonzero(pend)
        golpe = muestrear(P[idx[sub]] + (i * paso) * d[None, :], G, vol)
        som[idx[sub[golpe]]] = True
        pend[sub[golpe]] = False

    out = som.reshape(hs, ws).astype(np.uint8)
    if step > 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)

def comparar(pred, obs, valido):
    """IoU restringido a la region OBSERVABLE (ev >= 0)."""
    p, o = pred & valido, obs & valido
    inter = float((p & o).sum())
    return dict(area_pred=int(p.sum()), area_obs=int(o.sum()),
                frac_valida=float(valido.mean()),
                iou=inter / max(float((p | o).sum()), 1.0),
                cob_obs=inter / max(float(o.sum()), 1.0),
                cob_pred=inter / max(float(p.sum()), 1.0))


def superponer(frame, ocupado_sil, som_pred, som_obs, alfa=0.45):
    """Composicion RGB: rojo = columna, cian = sombra predicha, amarillo = obs."""
    import cv2
    vis = frame.astype(np.float32).copy()
    for m, col in ((som_pred & ~ocupado_sil, (60, 200, 255)),
                   (ocupado_sil, (255, 70, 70))):
        if m.any():
            vis[m] = (1 - alfa) * vis[m] + alfa * np.array(col, np.float32)
    cont, _ = cv2.findContours(som_obs.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    vis = np.ascontiguousarray(np.clip(vis, 0, 255).astype(np.uint8))
    cv2.drawContours(vis, cont, -1, (255, 235, 0), 2)
    return vis


def direcciones_perpendiculares(K, H_c_w, vec_sol, W, H):
    """Normales horizontales a las lineas de vista de la camara y del sol."""
    _, d = pixel_a_rayo(W / 2.0, H / 2.0, K, H_c_w)
    dc = d[0][:2] / np.linalg.norm(d[0][:2])
    ds = -np.asarray(vec_sol, float)[:2]
    ds = ds / np.linalg.norm(ds)
    return (np.array([-dc[1], dc[0]]), np.array([-ds[1], ds[0]]),
            dc, ds)

def tope_por_sombra(eje_xy, z_niveles, vec_sol, Z_interp, K, H_c_w, W, H,
                    som_obs, radios=None, n_muestras=8, frac_min=0.25,
                    paso=2.5, t_max=900.0):
    """Nivel mas alto cuya sombra cae dentro de la region observada."""
    z = np.asarray(z_niveles, float)
    nz = len(z)
    eje = np.asarray(eje_xy, float)
    if eje.ndim == 1:
        eje = np.repeat(eje[None, :], nz, axis=0)
    R = np.zeros(nz) if radios is None else np.nan_to_num(np.asarray(radios, float))

    ang = np.linspace(0, 2 * np.pi, n_muestras, endpoint=False)
    off = np.concatenate([np.zeros((1, 2)),
                          np.stack([np.cos(ang), np.sin(ang)], axis=1)])
    m = len(off)

    P = np.empty((nz * m, 3))
    for k in range(nz):
        rr = 0.5 * R[k]
        P[k*m:(k+1)*m, :2] = eje[k][None, :] + rr * off
        P[k*m:(k+1)*m, 2] = z[k]

    D = np.repeat((-np.asarray(vec_sol, float))[None, :], len(P), axis=0)
    Pg, tg = interseccion_rayos_dem(P, D, Z_interp, t_max=t_max,
                                       paso=paso, n_bisec=20)
    uv, delante = mundo_a_pixel(np.nan_to_num(Pg, nan=0.0), K, H_c_w)
    u, v = uv[:, 0], uv[:, 1]
    enc = (np.isfinite(tg) & delante &
           (u >= 0) & (u < W) & (v >= 0) & (v < H))
    ui = np.clip(np.round(u), 0, W - 1).astype(int)
    vi = np.clip(np.round(v), 0, H - 1).astype(int)

    dentro = np.zeros(len(P), bool)
    dentro[enc] = som_obs[vi[enc], ui[enc]]

    d_niv = dentro.reshape(nz, m)
    e_niv = enc.reshape(nz, m)
    n_enc = e_niv.sum(axis=1)
    frac = np.divide(d_niv.sum(axis=1), np.maximum(n_enc, 1), dtype=float)
    valido = (n_enc > 0) & (frac >= frac_min)

    if not valido.any():
        return nz - 1, float(z[-1]), False, True

    k_top = int(np.max(np.flatnonzero(valido)))
    limitado = bool(k_top + 1 < nz and n_enc[k_top + 1] == 0)
    return k_top, float(z[k_top]), limitado, False

def revolucion(G, ocupado, n_cam, n_sol, k_top=None, min_pts=3):
    """Cierra el hull suponiendo seccion circular por nivel."""
    P, idx, vz = G["P"], G["idx"], G["vz"]
    nz = len(vz)
    niv = idx[:, 2]
    if k_top is None:
        k_top = nz - 1

    R = np.zeros(nz)
    C = np.full((nz, 2), np.nan)
    M = np.stack([n_cam, n_sol])
    for k in range(min(k_top, nz - 1) + 1):
        s = ocupado & (niv == k)
        if s.sum() < min_pts:
            continue
        xy = P[s, :2]
        a, b = xy @ n_cam, xy @ n_sol
        R[k] = 0.5 * min(a.max() - a.min(), b.max() - b.min())
        C[k] = np.linalg.solve(M, np.array([0.5 * (a.min() + a.max()),
                                            0.5 * (b.min() + b.max())]))

    con = np.isfinite(C[:, 0])
    if con.sum() >= 3:
        cx = np.polyval(np.polyfit(vz[con], C[con, 0], 1), vz)
        cy = np.polyval(np.polyfit(vz[con], C[con, 1], 1), vz)
        eje = np.stack([cx, cy], axis=1)
    elif con.any():
        eje = np.repeat(np.nanmean(C[con], axis=0)[None, :], nz, axis=0)
    else:
        return np.zeros(len(P), bool), R, np.zeros((nz, 2))

    d = np.linalg.norm(P[:, :2] - eje[niv], axis=1)
    salida = ocupado & (d <= R[niv]) & (R[niv] > 0) & (niv <= k_top)
    return salida, R, eje

def altura_por_punta(som_obs, XM, YM, ZM, pivote, d_h, tan_e, W,
                     margen_borde=15, pct=98.0, anillo=3.0):
    """Altura implicada por la punta de la sombra (cota superior)."""
    if som_obs.sum() < 200:
        return None
    vv, uu = np.nonzero(som_obs)
    xs, ys, zs = XM[som_obs], YM[som_obs], ZM[som_obs]
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if ok.sum() < 50:
        return None

    rel = np.stack([xs[ok] - pivote[0], ys[ok] - pivote[1]], axis=1)
    proy = rel @ np.asarray(d_h, float)[:2]
    sel = proy >= np.percentile(proy, pct) - anillo

    d_hor = float(np.median(proy[sel]))
    dz = float(np.median(zs[ok][sel])) - float(pivote[2])

    H_img = som_obs.shape[0]
    u_p, v_p = uu[ok][sel], vv[ok][sel]
    truncada = bool(u_p.min() <= margen_borde or
                    u_p.max() >= W - 1 - margen_borde or
                    v_p.min() <= margen_borde or
                    v_p.max() >= H_img - 1 - margen_borde)
    return dict(h=dz + d_hor * tan_e, d_hor=d_hor, dz=dz, truncada=truncada)


def precalcular_corredor(XM, YM, ZM, vec_sol, K, H_c_w, W, H,
                         h_max=160.0, paso=6.0, step=4, verbose=True):
    """Tabla de busqueda del corredor de sombra, calculada una vez."""
    import time
    t0 = time.time()
    Xs, Ys, Zs = XM[::step, ::step], YM[::step, ::step], ZM[::step, ::step]
    hs, ws = Xs.shape
    hay = np.isfinite(Xs) & np.isfinite(Ys) & np.isfinite(Zs)
    P = np.stack([np.nan_to_num(Xs).ravel(), np.nan_to_num(Ys).ravel(),
                  np.nan_to_num(Zs).ravel()], axis=1)

    d = np.asarray(vec_sol, float)
    t_max = h_max / max(d[2], 1e-6)
    n = int(np.ceil(t_max / paso))

    idx = np.full((n, len(P)), -1, np.int32)
    for k in range(n):
        Q = P + ((k + 1) * paso) * d[None, :]
        uv, delante = mundo_a_pixel(Q, K, H_c_w)
        uv = np.nan_to_num(uv, nan=-1e9)
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ok = (delante & hay.ravel() &
              (u >= 0) & (u < W) & (v >= 0) & (v < H))
        idx[k, ok] = (v[ok] * W + u[ok]).astype(np.int32)

    if verbose:
        print(f"  corredor : {n} pasos x {len(P):,} px "
              f"({idx.nbytes/1e6:.0f} MB, {time.time()-t0:.1f} s)")
    return dict(idx=idx, forma=(hs, ws), step=int(step), W=int(W), H=int(H))

def corredor_de_gas(m_gas, CORR):
    """Terreno donde la sombra de la columna puede caer: pixeles cuyo rayo al sol cruza el
    cono de camara del gas.
    """
    import cv2
    g = m_gas.ravel()
    I = CORR["idx"]
    hit = (g[np.maximum(I, 0)] & (I >= 0)).any(axis=0)
    out = hit.reshape(CORR["forma"]).astype(np.uint8)
    if CORR["step"] > 1:
        out = cv2.resize(out, (CORR["W"], CORR["H"]),
                         interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)

def sombra_columna_geom(m_sombra, m_gas, m_estatica, CORR,
                        radio_estatica=5, radio_gas=3):
    """Sombra atribuible a la columna, por geometria en vez de conectividad."""
    import cv2
    def _k(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)

    est = cv2.dilate(m_estatica.astype(np.uint8), _k(radio_estatica)).astype(bool)
    gas = cv2.dilate(m_gas.astype(np.uint8), _k(radio_gas)).astype(bool)
    cor = corredor_de_gas(m_gas, CORR)

    salida = m_sombra & cor & ~gas & ~est
    return salida, dict(area_v6=int(m_sombra.sum()),
                        area_corredor=int(cor.sum()),
                        area_sobre_gas=int((m_sombra & gas).sum()),
                        area_en_estatica=int((m_sombra & cor & ~gas & est).sum()),
                        area_final=int(salida.sum()))

def altura_por_punta_eje(som_obs, XM, YM, ZM, pivote, d_h, tan_e, W,
                         eje=None, z_niveles=None, n_iter=2,
                         margen_borde=15, pct=98.0, anillo=3.0):
    """Altura por la punta de la sombra, descontando la deriva horizontal de la cima del
    penacho.
    """
    base = altura_por_punta(som_obs, XM, YM, ZM, pivote, d_h, tan_e, W,
                            margen_borde=margen_borde, pct=pct, anillo=anillo)
    if base is None:
        return None
    base["h_sin_correccion"] = base["h"]
    base["s"] = 0.0
    if eje is None or z_niveles is None:
        return base

    eje = np.asarray(eje, float)
    z = np.asarray(z_niveles, float)
    fin = np.isfinite(eje[:, 0]) & np.isfinite(eje[:, 1])
    if fin.sum() < 2:
        return base

    dh2 = np.asarray(d_h, float)[:2]
    h = base["h"]
    for _ in range(int(n_iter)):
        k = int(np.argmin(np.abs(z[fin] - (pivote[2] + h))))
        c = eje[fin][k]
        s = float((c - np.asarray(pivote, float)[:2]) @ dh2)
        h_new = base["dz"] + (base["d_hor"] - s) * tan_e
        if not np.isfinite(h_new):
            break
        h = h_new
    base["s"], base["h"] = s, h
    return base


def _caja(G):
    r = G["res"] / 2.0
    lo = np.array([G["vx"][0], G["vy"][0], G["vz"][0]]) - r
    hi = np.array([G["vx"][-1], G["vy"][-1], G["vz"][-1]]) + r
    return lo, hi


def _hg(cos_t, g):
    """Fase de Henyey-Greenstein normalizada a 1 en g=0."""
    g = float(np.clip(g, -0.9, 0.9))
    return (1.0 - g * g) / np.power(1.0 + g * g - 2.0 * g * cos_t, 1.5)


def _slab(O, D, lo, hi):
    """Interseccion rayo-caja alineada a ejes. Recorta la marcha al volumen."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo[None, :] - O) / D
        t2 = (hi[None, :] - O) / D
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    tmin = np.maximum(tmin, 0.0)
    return tmin, tmax, (tmax > tmin)


def campo_densidad(G, ocupado, suavizado=1.2, sigma_max=0.09):
    """Hull binario -> campo de densidad continuo."""
    from scipy.ndimage import gaussian_filter
    vol = denso(G, ocupado).astype(np.float32)
    if not vol.any():
        return vol
    if suavizado > 0:
        vol = gaussian_filter(vol, float(suavizado))
        m = float(vol.max())
        if m > 0:
            vol /= m
    return (vol * float(sigma_max)).astype(np.float32)


def muestrear_campo(P, G, campo):
    """Valor de un campo 3D en puntos UTM arbitrarios (vecino mas cercano)."""
    nx, ny, nz = campo.shape
    ix = np.rint((P[:, 0] - G["vx"][0]) / G["res"]).astype(np.int32)
    iy = np.rint((P[:, 1] - G["vy"][0]) / G["res"]).astype(np.int32)
    iz = np.rint((P[:, 2] - G["vz"][0]) / G["res"]).astype(np.int32)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) &
          (iz >= 0) & (iz < nz))
    out = np.zeros(len(P), np.float32)
    if ok.any():
        out[ok] = campo[ix[ok], iy[ok], iz[ok]]
    return out


def tau_solar(G, sigma, vec_sol, paso=None):
    """Profundidad optica DESDE cada voxel HACIA el sol."""
    if paso is None:
        paso = G["res"]
    ijk = np.argwhere(sigma > 1e-6)
    out = np.zeros_like(sigma)
    if len(ijk) == 0:
        return out

    P = np.stack([G["vx"][ijk[:, 0]], G["vy"][ijk[:, 1]],
                  G["vz"][ijk[:, 2]]], axis=1)
    d = np.asarray(vec_sol, float)
    lo, hi = _caja(G)
    _, tmax, _ = _slab(P, np.repeat(d[None, :], len(P), axis=0), lo, hi)
    tmax = np.nan_to_num(tmax, nan=0.0)

    acc = np.zeros(len(P), np.float32)
    for i in range(int(np.ceil(tmax.max() / paso)) if tmax.max() > 0 else 0):
        t = (i + 0.5) * paso
        act = t < tmax
        if not act.any():
            break
        acc[act] += muestrear_campo(P[act] + t * d[None, :], G, sigma)
    out[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = acc * paso
    return out


def color_gas_por_frame(cfg, mask_gas, frames, pct=50, normalizar=True,
                        min_px=200, verbose=True):
    """Color del penacho MEDIDO en el video, frame a frame."""
    out, crudos = {}, []
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for fi, small in lec.recorrer(sorted(int(f) for f in frames)):
            if small is None or fi not in mask_gas:
                continue
            m = np.asarray(mask_gas[fi]) > 0
            m = m[:small.shape[0], :small.shape[1]]
            if int(m.sum()) < min_px:
                continue
            px = small[:m.shape[0], :m.shape[1]][m].astype(np.float32)
            bgr = np.percentile(px, pct, axis=0)
            rgb = bgr[::-1] / 255.0
            crudos.append(rgb)
            if normalizar:
                mx = float(rgb.max())
                rgb = rgb / mx if mx > 1e-6 else rgb
            out[int(fi)] = [float(c) for c in rgb]
    if verbose and crudos:
        c = np.median(np.stack(crudos), axis=0) * 255.0
        print(f"  color del gas medido en {len(out)} frames | mediana cruda "
              f"R {c[0]:.0f} G {c[1]:.0f} B {c[2]:.0f}")
        if normalizar:
            n = c / max(c.max(), 1e-6)
            print(f"  tono normalizado (el render pone la intensidad): "
                  f"R {n[0]:.2f} G {n[1]:.2f} B {n[2]:.2f}")
    elif verbose:
        print("  [aviso] ningun frame con mascara suficiente; sin color medido")
    return out


def preparar_escena(cfg, ruta_calibracion=None, escala=None):
    """DEM, camara, sol y pivote de la escena."""
    escala = cfg.escala if escala is None else escala
    fd = dict(cfg.frame_data)

    print("DEM:")
    dem = leer_dem_ventana(cfg.dem, cfg.bounds_mina)
    assert str(dem["crs"]) == "EPSG:32719", f"CRS inesperado: {dem['crs']}"

    print("\nCamara:")
    K, H_c_w, FD = construir_camara(fd, escala=escala, dem_ctx=dem)
    W, H = FD["width"], FD["height"]

    if ruta_calibracion is None and os.path.exists(cfg.art_geometria):
        g = F0.cargar_geometria(cfg.art_geometria)
        fuente_geo = os.path.basename(cfg.art_geometria)
    else:
        cal = F0.cargar_calibracion(ruta_calibracion or cfg.calibracion)
        g = F0.resolver_geometria(cfg, W, H, calib=cal, verbose=False)
        fuente_geo = os.path.basename(ruta_calibracion or cfg.calibracion)

    pivot_frame = int(cfg.pivot_frame if cfg.pivot_frame is not None
                      else g["blast_frame"])
    if g.get("blast_frame") is not None and abs(g["blast_frame"] - pivot_frame) > 1:
        print(f"  [aviso] cfg.pivot_frame = {pivot_frame} y la calibracion dice "
              f"{g['blast_frame']}. Manda el de la config: es el que usaron las "
              f"fases anteriores, y mezclarlos corre el eje temporal.")
    pivot_px = (float(g["pivote_px_nat"][0]) * escala,
                float(g["pivote_px_nat"][1]) * escala)

    o, d = pixel_a_rayo(pivot_px[0], pivot_px[1], K, H_c_w)
    pts, t = interseccion_rayos_dem(o, d, dem["Z_interp"], t_max=5000.0,
                                    paso=5.0, n_bisec=30)
    if not np.isfinite(t[0]):
        raise RuntimeError("El rayo del pivote no intersecta el DEM.")
    pivote = pts[0]

    _cap = cv2.VideoCapture(cfg.video)
    try:
        fps = float(_cap.get(cv2.CAP_PROP_FPS))
    finally:
        _cap.release()
    ts0 = datetime.datetime.fromisoformat(cfg.ts_inicio_video).replace(
        tzinfo=ZoneInfo(cfg.tz_local))
    ts_det = ts0 + datetime.timedelta(seconds=pivot_frame / fps)
    sol = posicion_solar(ts_det, fd["lat_deg"], fd["lon_deg"])
    vec_sol = sol["vec_sol"]
    tan_e = np.tan(np.radians(sol["elevacion_deg"]))
    d_h = -vec_sol.copy(); d_h[2] = 0.0; d_h /= np.linalg.norm(d_h)

    n_cam, n_sol, dir_cam, dir_sol = direcciones_perpendiculares(
        K, H_c_w, vec_sol, W, H)
    sep = abs((np.degrees(np.arctan2(*dir_sol))
               - np.degrees(np.arctan2(*dir_cam))) % 360)

    print(f"\ngeometria: {fuente_geo}")
    print(f"pivote  : f{pivot_frame} | UTM E {pivote[0]:.2f} "
          f"N {pivote[1]:.2f} z {pivote[2]:.2f}")
    print(f"sol     : az {sol['azimut_deg']:.2f}  elev {sol['elevacion_deg']:.2f}")
    print(f"separacion entre vistas: {sep:.0f} deg  (90 seria el optimo)")

    return dict(dem=dem, K=K, H_c_w=H_c_w, FD=FD, W=W, H=H, escala=escala,
                pivot_frame=pivot_frame, pivot_px=pivot_px, pivote=pivote,
                origen=pivote.copy(), sol=sol, vec_sol=vec_sol, tan_e=tan_e,
                d_h=d_h, n_cam=n_cam, n_sol=n_sol, fps=fps, forma=(H, W))

def mapas_terreno(esc, dir_cache, usar_cache=True, paso=2.0, step=1):
    """XM, YM, ZM (un rayo por pixel) + auto-sombra del relieve. Se cachea."""
    ruta = os.path.join(
        dir_cache, f"terreno_{esc['W']}x{esc['H']}_p{paso:g}_s{step}.npz")
    if usar_cache and os.path.exists(ruta):
        z = np.load(ruta)
        XM, YM, ZM, auto = z["XM"], z["YM"], z["ZM"], z["AUTOSOMBRA"]
        print(f"cargado de cache: {os.path.basename(ruta)}")
    else:
        XM, YM, ZM = mapas_terreno_xyz(esc["K"], esc["H_c_w"], esc["W"], esc["H"],
                                       esc["dem"]["Z_interp"], paso=paso, step=step)
        auto = mapa_autosombra(XM, YM, ZM, esc["vec_sol"],
                               esc["dem"]["Z_interp"])
        os.makedirs(dir_cache, exist_ok=True)
        np.savez_compressed(ruta, XM=XM, YM=YM, ZM=ZM, AUTOSOMBRA=auto)
        print(f"guardado: {F0.ruta_corta(ruta)}")
    sin_terreno = ~np.isfinite(XM)
    prof, _ = profundidad_terreno(XM, YM, ZM, esc["H_c_w"])
    uv, _ = mundo_a_pixel(
        np.stack([esc["pivote"], esc["pivote"] + 100.0 * esc["d_h"]]),
        esc["K"], esc["H_c_w"])
    px_por_m = float(np.linalg.norm(uv[1] - uv[0]) / 100.0)
    print(f"  sin terreno {sin_terreno.mean()*100:.2f}%  |  "
          f"auto-sombra {auto.mean()*100:.2f}%  |  {px_por_m:.2f} px/m")
    return dict(XM=XM, YM=YM, ZM=ZM, autosombra=auto, sin_terreno=sin_terreno,
                prof_terreno=prof, px_por_m=px_por_m)

def corredor_sombra(esc, ter, dir_cache, usar_cache=True, h_max=160.0,
                    paso=6.0, step=4):
    """Tabla de busqueda del corredor solar (se cachea)."""
    ruta = os.path.join(dir_cache, f"corredor_h{h_max:.0f}_p{paso:.0f}_s{step}.npz")
    if usar_cache and os.path.exists(ruta):
        z = np.load(ruta)
        corr = dict(idx=z["idx"], forma=tuple(z["forma"]), step=int(z["step"]),
                    W=int(z["W"]), H=int(z["H"]))
        print(f"cargado de cache: {os.path.basename(ruta)}")
    else:
        corr = precalcular_corredor(ter["XM"], ter["YM"], ter["ZM"],
                                    esc["vec_sol"], esc["K"], esc["H_c_w"],
                                    esc["W"], esc["H"], h_max=h_max,
                                    paso=paso, step=step)
        os.makedirs(dir_cache, exist_ok=True)
        np.savez_compressed(ruta, idx=corr["idx"], forma=np.array(corr["forma"]),
                            step=corr["step"], W=corr["W"], H=corr["H"])
        print(f"guardado: {F0.ruta_corta(ruta)}")
    print(f"  memoria {corr['idx'].nbytes/1e6:.0f} MB | alcance {h_max:.0f} m "
          f"= {h_max/esc['tan_e']:.0f} m de sombra")
    return corr


def evidencia(frames, mask_gas, mask_som, sombra_est, corr, esc, ter,
              radio_estatica=5, radio_gas=3, verbose=True):
    """SOMBRA_COL y EVID por frame."""
    forma = esc["forma"]
    som_col, evid = {}, {}
    if verbose:
        print(f"  {'frame':>6} {'gas%':>6} {'som_col%':>9} {'ev:-1%':>8}")
    for f in frames:
        if f in mask_som:
            sc, _ = sombra_columna_geom(mask_som[f], mask_gas[f], sombra_est,
                                        corr, radio_estatica=radio_estatica,
                                        radio_gas=radio_gas)
        else:
            sc = np.zeros(forma, bool)
        ev = evidencia_sombra(sc, mask_gas[f], sombra_est, ter["autosombra"],
                              ter["sin_terreno"], radio_estatica=radio_estatica,
                              radio_gas=radio_gas)
        som_col[f], evid[f] = sc, ev
        if verbose:
            print(f"  {f:>6} {mask_gas[f].mean()*100:>6.2f} "
                  f"{sc.mean()*100:>9.2f} {(ev==-1).mean()*100:>8.2f}")
    vacios = [f for f in frames if som_col[f].sum() < 200]
    if vacios and verbose:
        print(f"  sin sombra utilizable: {vacios}")
    return som_col, evid


def libre_solar(G, PR, ev_som, estricto=False):
    """Voxeles que la evidencia solar declara vacios."""
    lib = np.zeros(len(G["P"]), bool)
    o = PR["ok_s"]
    v = ev_som[PR["vs"][o], PR["us"][o]]
    lib[o] = (v != 1) if estricto else (v == 0)
    return lib


def rellenar_columnas(G, occ, occ_base=None):
    """Reconecta el penacho al suelo rellenando cada columna vertical."""
    nx, ny, nz = G["forma"]
    idx = G["idx"]

    def a_col(m):
        v = np.zeros((nx, ny, nz), bool)
        i = idx[m]
        v[i[:, 0], i[:, 1], i[:, 2]] = True
        return v.reshape(nx * ny, nz)

    col = a_col(occ)
    hay = col.any(axis=1)
    if not hay.any():
        return occ
    z_techo = nz - 1 - np.argmax(col[:, ::-1], axis=1)
    z_piso = np.argmax(col, axis=1)
    if occ_base is not None:
        cb = a_col(occ_base)
        hb = cb.any(axis=1)
        z_piso = np.where(hb, np.argmax(cb, axis=1), z_piso)
    z_piso = np.minimum(z_piso, z_techo)

    zz = np.arange(nz)[None, :]
    nuevo = (hay[:, None] & (zz >= z_piso[:, None])
             & (zz <= z_techo[:, None])).reshape(nx, ny, nz)
    return nuevo[idx[:, 0], idx[:, 1], idx[:, 2]]


def reconstruir(G, PR, esc, mask_gas_f, evid_f, som_col_f=None, k_top=None,
                usar_sombra=True, res_vox=2.5, tallado="despues",
                estricto=True, adaptativo=True, min_retencion=0.25,
                min_altura=0.6, rellenar=True):
    """Un frame: envolvente + cierre por revolucion (+ tope por sombra)."""
    if tallado not in ("antes", "despues"):
        raise ValueError(f"tallado debe ser 'antes' o 'despues', no {tallado!r}")

    en_gas = np.zeros(len(G["P"]), bool)
    en_gas[PR["ok_c"]] = mask_gas_f[PR["vc"][PR["ok_c"]], PR["uc"][PR["ok_c"]]]
    diag = dict(n_gas=int(en_gas.sum()),
                frac_sin_info=float((~PR["ok_s"] & en_gas).sum()
                                    / max(int(en_gas.sum()), 1)))

    def envolvente(lib):
        occ = en_gas & ~lib if lib is not None else en_gas.copy()
        occ, _ = regularizar(G, occ, cerrar=1, abrir=1, solo_mayor=True)
        return occ

    occ_cam = envolvente(None)
    tope = None
    if k_top is None and som_col_f is not None and occ_cam.sum() >= 8:
        _, R0, eje0 = revolucion(G, occ_cam, esc["n_cam"], esc["n_sol"], k_top=None)
        k_top, zt, lim, sin_ev = tope_por_sombra(
            eje0, G["vz"], esc["vec_sol"], esc["dem"]["Z_interp"], esc["K"],
            esc["H_c_w"], esc["W"], esc["H"], som_col_f, radios=R0, paso=res_vox)
        tope = dict(k=k_top, z=zt, h=zt - esc["pivote"][2], limitado=lim,
                    sin_ev=sin_ev)

    def cerrar(occ, lib_desp, base=None):
        rev, _, _ = revolucion(G, occ, esc["n_cam"], esc["n_sol"], k_top=k_top)
        if lib_desp is not None:
            rev = rev & ~lib_desp
            if rellenar:
                rev = rellenar_columnas(G, rev, occ_base=base)
        rev, _ = regularizar(G, rev, cerrar=1, abrir=0, solo_mayor=True)
        return rev

    rev_cam = cerrar(occ_cam, None)
    if not usar_sombra:
        diag.update(uso_sombra=False, n_ocupado=int(occ_cam.sum()))
        return dict(env=occ_cam, rev=rev_cam, tope=tope, diag=diag,
                    modo="sin_sombra")

    intentos = [("estricto", True), ("permisivo", False)] if estricto \
        else [("permisivo", False)]
    if not adaptativo:
        intentos = intentos[:1]

    piso = min_retencion * max(int(rev_cam.sum()), 1)
    h_cam = (float(np.percentile(altura_sobre_terreno(G, rev_cam), 99))
             if rev_cam.sum() else 0.0)
    for nombre, est in intentos:
        lib = libre_solar(G, PR, evid_f, estricto=est)
        if tallado == "antes":
            occ = envolvente(lib)
            rev = cerrar(occ, None)
            if rellenar:
                rev = rellenar_columnas(G, rev, occ_base=rev_cam)
                rev, _ = regularizar(G, rev, cerrar=1, abrir=0, solo_mayor=True)
        else:
            occ = occ_cam
            rev = cerrar(occ, lib, base=rev_cam)
        h_r = (float(np.percentile(altura_sobre_terreno(G, rev), 99))
               if rev.sum() else 0.0)
        ok = rev.sum() >= piso and h_r >= min_altura * h_cam
        if ok or not adaptativo:
            diag.update(uso_sombra=True, estricto=bool(est),
                        n_libre=int(lib.sum()), n_ocupado=int(occ.sum()),
                        retencion=float(rev.sum() / max(int(rev_cam.sum()), 1)),
                        h_rel=float(h_r / max(h_cam, 1e-9)))
            return dict(env=occ, rev=rev, tope=tope, diag=diag, modo=nombre)

    diag.update(uso_sombra=False, n_ocupado=int(occ_cam.sum()),
                retencion=1.0, respaldo=True)
    return dict(env=occ_cam, rev=rev_cam, tope=tope, diag=diag,
                modo="sin_sombra")

def curva_altura(frames, alturas, fps, pivot_frame, ventana_mediana=7,
                 h_min=3.0, h_max=120.0):
    """Rellena huecos, extrapola con ley de potencia y suaviza."""
    from scipy.ndimage import median_filter
    t = (np.asarray(frames, float) - pivot_frame) / fps
    h = np.asarray(alturas, float)
    ok = np.isfinite(h)
    if ok.sum() < 5:
        raise RuntimeError("Muy pocas alturas medidas; revisa som_col% en la evidencia.")
    m = ok & (t > 0)
    p = np.polyfit(np.log(t[m] + 1.0), np.log(h[m]), 1)
    h_fit = np.exp(np.polyval(p, np.log(np.maximum(t, 0.0) + 1.0)))
    est = h.copy()
    i0, i1 = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
    hueco = ~ok
    hueco[:i0] = False; hueco[i1 + 1:] = False
    if hueco.any():
        est[hueco] = np.interp(t[hueco], t[ok], h[ok])
    est[:i0] = h_fit[:i0]
    est[i1 + 1:] = h[i1]
    suave = np.clip(median_filter(est, size=ventana_mediana, mode="nearest"),
                    h_min, h_max)
    print(f"  ley de potencia: h = {np.exp(p[1]):.2f}·(t+1)^{p[0]:.2f}")
    print(f"  altura: {suave.min():.1f} – {suave.max():.1f} m")
    return t, suave, dict(A=float(np.exp(p[1])), n=float(p[0]))

def altura_limite_cuadro(esc, ter):
    """Altura sobre la cual la sombra sale del cuadro (cota de validez)."""
    W, H = esc["W"], esc["H"]
    uv, _ = mundo_a_pixel(
        np.stack([esc["pivote"], esc["pivote"] + 100.0 * esc["d_h"]]),
        esc["K"], esc["H_c_w"])
    dv = uv[1] - uv[0]
    n = float(np.linalg.norm(dv))
    if not np.isfinite(n) or n < 1e-9:
        return float("nan")
    du_, dv_ = dv / n
    u0, v0 = float(esc["pivot_px"][0]), float(esc["pivot_px"][1])

    ts = []
    if du_ > 1e-9:
        ts.append((W - 1 - u0) / du_)
    elif du_ < -1e-9:
        ts.append(-u0 / du_)
    if dv_ > 1e-9:
        ts.append((H - 1 - v0) / dv_)
    elif dv_ < -1e-9:
        ts.append(-v0 / dv_)
    ts = [t for t in ts if np.isfinite(t) and t > 0]
    if not ts or ter["px_por_m"] <= 0:
        return float("nan")
    return float(min(ts) / ter["px_por_m"] * esc["tan_e"])

def revisar_grilla(G, esc, mask_gas, frames, verbose=True):
    """Avisa si el penacho no cabe en la grilla de voxeles."""

    uv, ok = mundo_a_pixel(G["P"], esc["K"], esc["H_c_w"])
    W, H = esc["W"], esc["H"]
    huella = np.zeros((H, W), bool)
    u = np.clip(uv[ok, 0].astype(int), 0, W - 1)
    v = np.clip(uv[ok, 1].astype(int), 0, H - 1)
    huella[v, u] = True
    huella = np.asarray(
        cv2.dilate(huella.astype(np.uint8),
                                 np.ones((9, 9), np.uint8)) > 0)
    peor, peor_f = 0.0, None
    for f in frames:
        m = np.asarray(mask_gas[f]) > 0
        if not m.any():
            continue
        fuera = float((m & ~huella).sum()) / float(m.sum())
        if fuera > peor:
            peor, peor_f = fuera, f
    if verbose:
        if peor > 0.02:
            print(f"  [OJO] en f{peor_f} el {peor*100:.1f} % de la mascara de gas "
                  f"cae FUERA de la huella de la grilla.")
            print(f"        sube RADIO_XY (ahora "
                  f"{np.ptp(G['P'][:, :2], axis=0).max()/2:.0f} m) "
                  f"o Z_REL_MAX; si no, el hull sale recortado y parece correcto.")
        else:
            print(f"  grilla suficiente: como maximo {peor*100:.1f} % de la "
                  f"mascara queda fuera (f{peor_f}).")
    return dict(frac_fuera=peor, frame=peor_f)


def video_validacion(cfg, esc, ter, G, PR, mask_gas, evid, som_col, frames,
                     ruta_salida, fps, k_top=None, hulls=None, alturas=None,
                     usar_sombra=False, res_vox=2.5, step_sombra=4,
                     pivot_frame=None, alfa=0.45, leyenda=True, verbose=True):
    """Video de validacion: la reconstruccion superpuesta al frame real."""
    import time

    W, H = esc["W"], esc["H"]
    pf = esc["pivot_frame"] if pivot_frame is None else pivot_frame
    frames = [int(f) for f in frames]
    vw, (Wv, Hv) = F0.escritor_video(ruta_salida, fps, W, H)

    F = cv2.FONT_HERSHEY_SIMPLEX
    ESC_T = max(0.40, 0.55 * Wv / 960.0)
    GRU = max(1, int(round(ESC_T * 2)))
    DY = int(round(26 * ESC_T / 0.55))
    MX = int(round(15 * ESC_T / 0.55))
    BLA, NEG, AMA = (255, 255, 255), (0, 0, 0), (0, 235, 255)
    ROJ, CIA, NAR = (70, 70, 255), (255, 200, 60), (0, 165, 255)

    def texto(img, s, xy, col=BLA):
        cv2.putText(img, s, xy, F, ESC_T, NEG, GRU + 2, cv2.LINE_AA)
        cv2.putText(img, s, xy, F, ESC_T, col, GRU, cv2.LINE_AA)

    def ancho(s):
        return cv2.getTextSize(s, F, ESC_T, GRU)[0][0]

    t0, n_esc, n_vacio = time.time(), 0, 0
    with F0.LectorVideo(cfg.video, cfg.escala) as lec:
        for fi, small in lec.recorrer(frames):
            if small is None:
                continue
            small = small[:Hv, :Wv]
            mg = np.asarray(mask_gas[fi])[:Hv, :Wv] > 0 if fi in mask_gas \
                else np.zeros((Hv, Wv), bool)
            sc = np.asarray(som_col[fi])[:Hv, :Wv] if fi in som_col \
                else np.zeros((Hv, Wv), bool)

            if hulls is not None and fi in hulls:
                occ = hulls[fi]
            elif fi in evid:
                r = reconstruir(G, PR, esc, mask_gas[fi], evid[fi],
                                som_col_f=None if k_top else som_col.get(fi),
                                k_top=None if k_top is None else k_top.get(fi),
                                usar_sombra=usar_sombra, res_vox=res_vox)
                occ = r["rev"]
            else:
                occ = None

            vis = small[:, :, ::-1].copy()
            iou_s = iou_h = float("nan")
            if occ is not None and occ.sum():
                sil = silueta(G, occ, esc["K"], esc["H_c_w"], Wv, Hv)
                sp = sombra_proyectada(G, occ, ter["XM"][:Hv, :Wv],
                                       ter["YM"][:Hv, :Wv], ter["ZM"][:Hv, :Wv],
                                       esc["vec_sol"], step=step_sombra,
                                       paso=res_vox * 0.6)
                vis = superponer(vis, sil, sp, sc, alfa=alfa)
                un = float((sil | mg).sum())
                iou_h = float((sil & mg).sum()) / un if un else float("nan")
                if fi in evid:
                    iou_s = comparar(sp, sc, evid[fi][:Hv, :Wv] >= 0)["iou"]
            else:
                n_vacio += 1
            vis = np.ascontiguousarray(vis[:, :, ::-1])

            y = DY
            texto(vis, f"f{fi}   t {(fi - pf) / fps:+6.2f} s", (MX, y))
            h_use = None if alturas is None else alturas.get(fi)
            if h_use is not None and np.isfinite(h_use):
                y += DY
                texto(vis, f"altura usada {h_use:5.1f} m", (MX, y))
            ap = altura_por_punta(sc, ter["XM"][:Hv, :Wv], ter["YM"][:Hv, :Wv],
                                  ter["ZM"][:Hv, :Wv], esc["pivote"], esc["d_h"],
                                  esc["tan_e"], Wv) if sc.sum() >= 200 else None
            y += DY
            if ap is None:
                texto(vis, "sin sombra utilizable (<200 px)", (MX, y), NAR)
            else:
                texto(vis, f"punta de sombra {ap['h']:5.1f} m", (MX, y),
                      NAR if ap["truncada"] else BLA)
                if ap["truncada"]:
                    y += DY
                    texto(vis, "PUNTA EN EL BORDE -> cota inferior", (MX, y), NAR)
            texto(vis, f"IoU silueta {iou_h:.2f}   IoU sombra {iou_s:.2f}",
                  (MX, Hv - 2 * DY))
            if leyenda:
                cw = int(round(18 * ESC_T / 0.55))
                x = MX
                for col, et in ((ROJ, "hull proyectado"),
                                (CIA, "sombra predicha"),
                                (AMA, "sombra observada")):
                    paso_x = cw + int(round(6 * ESC_T / 0.55)) + ancho(et)
                    if x + paso_x > Wv - MX:
                        break
                    cv2.rectangle(vis, (x, Hv - DY - cw // 2),
                                  (x + cw, Hv - DY + cw // 2), col, -1)
                    texto(vis, et, (x + cw + int(round(6 * ESC_T / 0.55)),
                                    Hv - DY + cw // 3))
                    x += paso_x + int(round(20 * ESC_T / 0.55))

            vw.write(vis)
            n_esc += 1
            if verbose and n_esc % 20 == 0:
                el = time.time() - t0
                print(f"  {n_esc}/{len(frames)}  {el:5.0f} s  "
                      f"(ETA {el / n_esc * (len(frames) - n_esc):5.0f} s)", end="\r")
    vw.release()
    if verbose:
        print(f"\n[video] {F0.ruta_corta(ruta_salida)}  |  {n_esc} frames en "
              f"{time.time() - t0:.0f} s" +
              (f"  |  {n_vacio} sin hull" if n_vacio else ""))
    return ruta_salida


_PLANTILLA_BLENDER = r'''"""
blender_animar_gas.py - monta la escena de la Fase 2 dentro de Blender.
=======================================================================

La Fase 2 (F2-10) lo escribe junto a los archivos que consume:

    resultados/2_reconstruccion3d/
        gas_secuencia.glb    un objeto por keyframe, 'hull_fXXXXX'
        terreno.glb          el DEM texturizado con la ortofoto
        camara_dron.json     pose de camara, sol, fps, keyframes, alturas
        serie_columna.csv    la tabla de altura y volumen (no se usa aqui)

Uso
---
1. Abre Blender, pestaña Scripting, Open -> este archivo.
2. Run Script (Alt+P).

Los archivos se buscan en la carpeta de este script, asi que la carpeta
2_reconstruccion3d se puede mover o copiar a otro equipo. Si se ejecuta desde un
texto que no esta guardado en disco, se usa la ruta de la corrida que lo genero.

Que hace
--------
- importa terreno y gas, cada uno en su coleccion
- crea la camara con la pose e intrinsecos reales del dron
- crea el sol con el azimut y la elevacion del instante de la tronadura
- anima la visibilidad de los hulls para que la columna crezca en el tiempo
- deja la escena en el mismo sistema de coordenadas que la Fase 2:
      +X = Este   +Y = Norte   +Z = arriba,  origen en `origen_utm`

Convenciones que hay que respetar y por que
-------------------------------------------
`exportar_hulls_glb` escribe (E, U, -N) porque el importador de glTF de Blender
aplica (x, y, z)_gltf -> (x, -z, y)_blender. Las dos conversiones se cancelan y
las mallas quedan en (Este, Norte, arriba). En cambio `posicion_blender` del
JSON ya viene en (E, N, U) directo, sin swizzle: no lo conviertas otra vez.

`R_c_w` esta en convencion OpenCV (+X derecha, +Y ABAJO, +Z adelante). La camara
de Blender es +X derecha, +Y arriba, -Z adelante. La conversion es
R_blender = R_c_w @ diag(1, -1, -1); sin eso la camara queda de cabeza y
mirando al lado contrario.

Probado contra el esquema de `exportar_camara_json` y `exportar_hulls_glb`.
"""

import json
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════
def _carpeta_script():
    try:
        t = bpy.context.space_data.text
        if t is not None and t.filepath:
            return os.path.dirname(bpy.path.abspath(t.filepath))
    except AttributeError:
        pass
    return None


RUTA_SALIDAS = _carpeta_script() or r"@@RUTA_SALIDAS@@"

LIMPIAR_ESCENA = True     # borra todo antes de importar (recomendado al re-correr)
CARGAR_TERRENO = True     # el terreno son ~61 MB: apagalo para iterar rapido
COLOR_GAS      = (0.59, 0.58, 0.61)
MOSTRAR_ALTURA = True     # texto 3D con la altura del frame
ALTO_TEXTO     = 0.045    # alto del texto como fraccion del alto del cuadro
MARGEN_TEXTO   = 0.045    # margen desde la esquina, misma fraccion
SOMBRAS_SOL    = True

# ── DOS SOLES (light linking, Cycles en Blender 4.0 o superior) ───────────
# Un solo sol hace dos cosas que se estorban: proyecta la sombra de la nube
# sobre el terreno y tambien sombrea la nube por dentro, que es lo que la
# oscurece. Con DOS_SOLES = True se separan:
#   sol_terreno : ilumina SOLO el terreno y la nube le hace sombra
#   sol_gas     : ilumina SOLO el gas y no proyecta sombra
# Misma direccion y energia en ambos. Si la version no tiene light linking,
# se vuelve a un solo sol.
DOS_SOLES      = True

# "volumen"    : el hull se usa como CONTENEDOR y el gas se renderiza como
#                medio participativo. Es lo que hay que usar para un penacho:
#                tiene interior, se ve desde cualquier angulo y proyecta
#                sombra volumetrica. Necesita Cycles para verse bien; EEVEE
#                Next (4.2+) lo soporta con menos calidad.
# "superficie" : material transparente sobre la cascara. Mas rapido, pero como
#                una malla es solo una SUPERFICIE, al girar la camara se ve
#                una lamina con vacio detras, y el orden de las caras
#                transparentes en EEVEE hace que a veces desaparezca la cara
#                trasera. Es el modo que producia el efecto de "capas".
MODO_GAS       = "superficie"

# ── MOTOR ─────────────────────────────────────────────────────────────────
# "EEVEE"  : tiempo real. Es lo que hay que usar para NAVEGAR y reproducir la
#            animacion. Cycles en el viewport es progresivo: re-renderiza cada
#            frame desde cero acumulando muestras, asi que al reproducir se ve
#            pixelado siempre, por muchas muestras que se le bajen. No es un
#            ajuste mal puesto, es como funciona.
# "CYCLES" : para el render final. Mejor volumetria, mucho mas lento.
MOTOR          = "CYCLES"

# ── DENSIDAD ──────────────────────────────────────────────────────────────
# La densidad de un volumen en Blender es por METRO. El penacho mide decenas
# de metros, asi que se pide la profundidad optica TAU de lado a lado y la
# densidad se deriva del tamano real de las mallas.
TAU_GAS        = 3.0      # 0.6 tenue | 1.2 humo denso que deja pasar luz |
                          # 2.5 opaco. Por encima de ~2 el nucleo se oscurece
DENSIDAD_GAS   = None     # None = derivar de TAU_GAS y del tamano real

# ── POR QUE SE VEIA NEGRA ─────────────────────────────────────────────────
# El humo real se ve claro por DISPERSION MULTIPLE: la luz rebota muchas veces
# dentro de la nube. Cycles trae `volume_bounces = 0` de fabrica, o sea CERO
# rebotes dentro del volumen: solo llega la luz directa del sol, que aqui esta
# a 18.6 grados y entra de refilon. Con eso cualquier nube densa sale negra.
# Se sube a REBOTES_VOLUMEN. Y en EEVEE el rango volumetrico por defecto llega
# a 100 m; esta escena tiene cientos, asi que el volumen ni aparecia.
REBOTES_VOLUMEN = 4

# Ademas se le da al gas una EMISION propia con su color medido. Es un atajo
# -el humo no emite luz- pero garantiza que nunca salga negro y hace que
# Cycles converja mucho antes, porque la emision no necesita encontrar una
# fuente de luz por muestreo. Ponlo en 0.0 si quieres solo fisica.
EMISION_GAS    = 0.20
COLOR_DEL_VIDEO = True    # usar el color medido en el video, si el JSON lo trae
ALFA_GAS       = 1.0      # solo en modo superficie

MUESTRAS_VIEWPORT = 32
MUESTRAS_RENDER   = 128

# ── POR QUE NO SE VEIA LA SOMBRA ──────────────────────────────────────────
# Una sombra es CONTRASTE entre lo iluminado y lo no iluminado. Al subir el
# cielo a 1.2 para que el penacho no saliera negro, la luz difusa quedo casi
# tan fuerte como el sol: todo iluminado por igual y las sombras se borran.
# En luz diurna real el sol directo es varias veces la difusa del cielo.
# El piso de brillo del penacho ahora lo pone EMISION_GAS, asi que el cielo
# puede volver a un valor realista y la sombra reaparece.
ENERGIA_SOL       = 5.0    # sol directo
LUZ_AMBIENTE      = 0.35   # cielo difuso. Razon sol:cielo ~14:1, como en
                           # exterior despejado. Subirlo aplana las sombras

# Blender 4.x usa AgX como transformacion de vista: comprime mucho el rango y
# aplana el contraste, que para una figura tecnica no es lo que se quiere.
# "Standard" muestra los valores tal cual. Pon "AgX" si prefieres el look
# fotografico.
TRANSFORMACION_VISTA = "Standard"

# ══════════════════════════════════════════════════════════════════════════

RUTA_GLB = os.path.join(RUTA_SALIDAS, "gas_secuencia.glb")
RUTA_TER = os.path.join(RUTA_SALIDAS, "terreno.glb")
RUTA_CAM = os.path.join(RUTA_SALIDAS, "camara_dron.json")

for _r in (RUTA_GLB, RUTA_CAM):
    if not os.path.exists(_r):
        raise FileNotFoundError(
            f"No esta {_r}.\n"
            f"    Corre la celda C9 del notebook de la Fase 2 primero, y revisa "
            f"que RUTA_SALIDAS apunte a la carpeta correcta.")

with open(RUTA_CAM, encoding="utf-8") as fh:
    CAM = json.load(fh)


# ── utilidades ────────────────────────────────────────────────────────────
def limpiar():
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
    except RuntimeError as e:
        print(f"[aviso] no se pudo limpiar con operadores ({e}); se borra a mano")
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
    for bloque in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                   bpy.data.lights, bpy.data.curves, bpy.data.images):
        for x in list(bloque):
            if x.users == 0:
                bloque.remove(x)


def coleccion(nombre):
    c = bpy.data.collections.get(nombre)
    if c is None:
        c = bpy.data.collections.new(nombre)
        bpy.context.scene.collection.children.link(c)
    return c


def importar(ruta, col):
    """Importa un GLB y mueve lo nuevo a `col`. Devuelve los objetos nuevos."""
    antes = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=ruta)
    nuevos = [o for o in bpy.data.objects if o not in antes]
    for o in nuevos:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    return nuevos


def srgb_a_lineal(c):
    """Blender trabaja en lineal; lo medido en el video esta en sRGB."""
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def tamano_tipico(objs):
    """Cuerda media del penacho, en metros: mediana de la diagonal horizontal
    de los bounding boxes. Es la escala sobre la que se integra la densidad."""
    d = []
    for o in objs:
        b = [Vector(v) for v in o.bound_box]
        xs = [v.x for v in b]; ys = [v.y for v in b]; zs = [v.z for v in b]
        dx, dy, dz = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
        if max(dx, dy, dz) > 1e-6:
            d.append((dx * dy * dz) ** (1.0 / 3.0))     # lado del cubo equivalente
    if not d:
        return 50.0
    d.sort()
    return float(d[len(d) // 2])


def preparar_terreno(objs):
    """Deja el terreno en condiciones de RECIBIR la sombra del penacho.

    La ortofoto ya trae la iluminacion real horneada, y el importador de glTF
    a veces la deja con especular o con algo de emision: sobre una superficie
    asi la sombra se lava o directamente no se nota. Se fuerza un difuso puro.
    """
    n = 0
    for o in objs:
        if getattr(o, "type", "") != "MESH":
            continue
        o.visible_shadow = True
        for ranura in o.material_slots:
            m = ranura.material
            if m is None or not m.use_nodes:
                continue
            for nodo in m.node_tree.nodes:
                if nodo.type != "BSDF_PRINCIPLED":
                    continue
                for nom, val in (("Roughness", 1.0), ("Metallic", 0.0),
                                 ("Specular IOR Level", 0.0), ("Specular", 0.0),
                                 ("Emission Strength", 0.0)):
                    ent = nodo.inputs.get(nom)
                    if ent is not None and not ent.is_linked:
                        ent.default_value = val
                n += 1
    if n:
        print(f"[terreno] {n} materiales pasados a difuso puro para que la "
              f"sombra se note")


def material_gas():
    """Material del penacho.

    El hull que exporta la Fase 2 es MACIZO en voxeles, pero `hull_a_malla`
    lo convierte a malla con marching cubes, y una malla es una superficie:
    por dentro no hay nada. Si encima se le pone un material transparente, al
    girar la camara se ve una lamina con vacio detras. La solucion no es
    cambiar la geometria sino decirle a Blender que el interior es materia:
    el material de volumen usa la malla como CONTENEDOR e integra densidad
    dentro de el.
    """
    m = bpy.data.materials.get("gas_columna")
    if m is None:
        m = bpy.data.materials.new("gas_columna")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (300, 0)

    if MODO_GAS == "volumen":
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.name = "volumen_gas"
        vol.location = (0, 0)
        vol.inputs["Color"].default_value = (*COLOR_GAS, 1.0)
        vol.inputs["Density"].default_value = DENSIDAD
        if "Anisotropy" in vol.inputs:
            vol.inputs["Anisotropy"].default_value = 0.3
        # Emision con el mismo color: piso de brillo para que el penacho no
        # pueda salir negro, y ademas hace converger a Cycles mucho antes.
        for nom, val in (("Emission Strength", EMISION_GAS),
                         ("Emission Color", (*COLOR_GAS, 1.0))):
            if nom in vol.inputs:
                vol.inputs[nom].default_value = val
        nt.links.new(vol.outputs["Volume"], salida.inputs["Volume"])
        # la superficie queda SIN conectar: la malla es solo el contenedor
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (*COLOR_GAS, 1.0)
        bsdf.inputs["Alpha"].default_value = ALFA_GAS
        for nom, val in (("Roughness", 1.0), ("Specular IOR Level", 0.0),
                         ("Specular", 0.0), ("Metallic", 0.0)):
            if nom in bsdf.inputs:
                bsdf.inputs[nom].default_value = val
        nt.links.new(bsdf.outputs["BSDF"], salida.inputs["Surface"])
        # La transparencia en EEVEE cambio de API en Blender 4.2.
        if hasattr(m, "surface_render_method"):      # 4.2+
            m.surface_render_method = "BLENDED"
        elif hasattr(m, "blend_method"):             # 3.x - 4.1
            m.blend_method = "BLEND"
            if hasattr(m, "shadow_method"):
                m.shadow_method = "HASHED"
        if hasattr(m, "use_backface_culling"):
            m.use_backface_culling = False
    return m


# ══════════════════════════════════════════════════════════════════════════
# 1. ESCENA
# ══════════════════════════════════════════════════════════════════════════
if LIMPIAR_ESCENA:
    limpiar()

esc = bpy.context.scene
fps_video = float(CAM.get("fps", 30.0))
esc.render.fps = int(round(fps_video))
esc.render.resolution_x = int(CAM["ancho_px"])
esc.render.resolution_y = int(CAM["alto_px"])
esc.render.resolution_percentage = 100

def poner_motor(nombre):
    """EEVEE cambio de identificador en 4.2 (BLENDER_EEVEE -> ..._NEXT)."""
    if not nombre:
        return esc.render.engine
    cands = (["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"] if nombre.upper() == "EEVEE"
             else [nombre])
    for c in cands:
        try:
            esc.render.engine = c
            return c
        except TypeError:
            continue
    print(f"[aviso] no se pudo poner '{nombre}'; se deja {esc.render.engine}")
    return esc.render.engine


poner_motor(MOTOR)
print(f"[motor] {esc.render.engine}  |  Blender {bpy.app.version_string}")

if esc.render.engine == "CYCLES":
    cy = esc.cycles
    cy.preview_samples = MUESTRAS_VIEWPORT
    cy.samples = MUESTRAS_RENDER
    # ESTE es el ajuste que quitaba la luz de dentro de la nube.
    if hasattr(cy, "volume_bounces"):
        cy.volume_bounces = REBOTES_VOLUMEN
        cy.max_bounces = max(getattr(cy, "max_bounces", 12), REBOTES_VOLUMEN + 4)
        print(f"[cycles] volume_bounces {cy.volume_bounces} "
              f"(de fabrica viene en 0: sin dispersion multiple el humo denso "
              f"sale NEGRO)")
    for attr, val in (("use_preview_denoising", True), ("use_denoising", True),
                      ("preview_denoising_start_sample", 1)):
        if hasattr(cy, attr):
            setattr(cy, attr, val)
    print(f"[cycles] viewport {MUESTRAS_VIEWPORT} muestras con denoise | "
          f"render {MUESTRAS_RENDER}")
    print("         OJO: el viewport de Cycles es progresivo. Para NAVEGAR y "
          "reproducir\n         usa MOTOR='EEVEE'; deja Cycles para el render "
          "final.")
elif esc.render.engine.startswith("BLENDER_EEVEE"):
    ee = esc.eevee
    # El rango volumetrico de EEVEE va de 0.1 a 100 m por defecto. Esta escena
    # tiene cientos de metros: fuera de ese rango el volumen no se dibuja.
    lejos = max(2000.0, 3.0 * Vector(CAM["posicion_blender"]).length)
    for attr, val in (("volumetric_start", 0.1), ("volumetric_end", lejos),
                      ("volumetric_tile_size", "2"), ("volumetric_samples", 128),
                      ("use_volumetric_shadows", True),
                      ("volumetric_shadow_samples", 64),
                      ("use_shadows", True), ("shadow_ray_count", 2),
                      ("shadow_step_count", 6),
                      ("taa_samples", 16), ("taa_render_samples", 64)):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except (TypeError, ValueError):
                pass
    print(f"[eevee] rango volumetrico 0.1 - {lejos:.0f} m "
          f"(de fabrica llega a 100 m y el penacho quedaba fuera)")
    print(f"[eevee] tiempo real: navegar y reproducir sin ruido progresivo")
    print("[eevee] OJO con la SOMBRA del penacho sobre el terreno: EEVEE la\n"
          "        aproxima y a veces no la dibuja. Es el precio del tiempo\n"
          "        real. Para verla bien pon MOTOR = 'CYCLES' y mira UN frame\n"
          "        (no reproduzcas): ahi la sombra volumetrica es exacta.")

# Cielo: un volumen iluminado SOLO por un sol a 18.6 grados queda oscuro por
# dentro. La luz difusa del cielo es fisica, no maquillaje.
if esc.world is None:
    esc.world = bpy.data.worlds.new("mundo")
esc.world.use_nodes = True
_bg = esc.world.node_tree.nodes.get("Background")
if _bg is not None:
    _bg.inputs["Color"].default_value = (0.55, 0.65, 0.82, 1.0)
    _bg.inputs["Strength"].default_value = LUZ_AMBIENTE
print(f"[luz] sol {ENERGIA_SOL:.1f} | cielo {LUZ_AMBIENTE:.2f} | "
      f"razon {ENERGIA_SOL/max(LUZ_AMBIENTE,1e-6):.0f}:1")
print("      una sombra es CONTRASTE: si el cielo se acerca al sol, se borra.")

if TRANSFORMACION_VISTA:
    try:
        esc.view_settings.view_transform = TRANSFORMACION_VISTA
        print(f"[vista] transformacion {esc.view_settings.view_transform} "
              f"(AgX aplana el contraste de las sombras)")
    except TypeError:
        print(f"[aviso] '{TRANSFORMACION_VISTA}' no existe en esta version; "
              f"se deja {esc.view_settings.view_transform}")

col_gas = coleccion("gas")
col_ter = coleccion("terreno")

# ══════════════════════════════════════════════════════════════════════════
# 2. IMPORTAR
# ══════════════════════════════════════════════════════════════════════════
if CARGAR_TERRENO and os.path.exists(RUTA_TER):
    ter = importar(RUTA_TER, col_ter)
    print(f"[terreno] {len(ter)} objetos")
    preparar_terreno(ter)
elif CARGAR_TERRENO:
    print(f"[terreno] no esta {RUTA_TER}; se sigue sin el")

objs = importar(RUTA_GLB, col_gas)
hulls = sorted((o for o in objs if o.name.startswith("hull_f")),
               key=lambda o: o.name)
if not hulls:
    # trimesh a veces renombra los nodos: se toma todo lo importado, en orden
    hulls = objs
    print("[aviso] ningun objeto se llama 'hull_fXXXXX'; se usa el orden de "
          "importacion. Revisa que el GLB sea el de exportar_hulls_glb.")
print(f"[gas] {len(hulls)} hulls importados  |  material en modo {MODO_GAS}")
if MODO_GAS == "volumen":
    print("      el gas es un medio participativo dentro de la malla: tiene "
          "interior y se\n      ve igual desde cualquier angulo. Si sale muy "
          "tenue sube DENSIDAD_GAS.")

def normales_hacia_afuera(objs):
    """Cycles usa la orientacion de las caras para saber si un rayo entra o
    sale del volumen. Con las normales hacia adentro el gas queda 'fuera' de
    la malla. Se invierte cada malla cuyo volumen con signo sea negativo."""
    n = 0
    for o in objs:
        if getattr(o, "type", "") != "MESH":
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        if bm.calc_volume(signed=True) < 0:
            bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
            bm.to_mesh(o.data)
            o.data.update()
            n += 1
        bm.free()
    print(f"[gas] {n} de {len(objs)} mallas tenian las normales hacia adentro; "
          f"se invirtieron")


normales_hacia_afuera(hulls)

TAM = tamano_tipico(hulls)
DENSIDAD = (TAU_GAS / max(TAM, 1e-6)) if DENSIDAD_GAS is None else DENSIDAD_GAS
print(f"[densidad] penacho tipico {TAM:.0f} m  ->  densidad {DENSIDAD:.4f} /m "
      f"(tau {TAU_GAS:.1f} de lado a lado)")

mat = material_gas()
for o in hulls:
    o.data.materials.clear()
    o.data.materials.append(mat)

# ── color medido en el video ──────────────────────────────────────────────
_COL = CAM.get("color_gas_srgb") if COLOR_DEL_VIDEO else None
if _COL:
    _nodo = mat.node_tree.nodes.get("volumen_gas")
    if _nodo is None:
        _cs = None
        _cs0 = {int(k): v for k, v in _COL.items()}
        _prom0 = [sum(c[i] for c in _cs0.values()) / len(_cs0) for i in range(3)]
        for _n in mat.node_tree.nodes:
            if _n.type == "BSDF_PRINCIPLED":
                _n.inputs["Base Color"].default_value = (
                    *(srgb_a_lineal(c) for c in _prom0), 1.0)
        print(f"[color] modo superficie: color medio del video "
              f"({_prom0[0]:.2f}, {_prom0[1]:.2f}, {_prom0[2]:.2f})")
    else:
        _cs = {int(k): v for k, v in _COL.items()}
        _prom = [sum(c[i] for c in _cs.values()) / len(_cs) for i in range(3)]
        print(f"[color] medido en el video: sRGB medio "
              f"({_prom[0]:.2f}, {_prom[1]:.2f}, {_prom[2]:.2f}) "
              f"-> se anima por frame")
else:
    _cs = None
    if COLOR_DEL_VIDEO:
        print("[color] EL JSON NO TRAE 'color_gas_srgb'. Vuelve a correr la "
              "celda C9 del\n        notebook: es la que mide el color del gas "
              "en el video y lo escribe.\n        Mientras tanto se usa "
              f"COLOR_GAS = {COLOR_GAS}.")
    else:
        print(f"[color] COLOR_DEL_VIDEO=False; se usa COLOR_GAS del script")

# ══════════════════════════════════════════════════════════════════════════
# 3. CAMARA  (pose e intrinsecos reales del dron)
# ══════════════════════════════════════════════════════════════════════════
cam_dat = bpy.data.cameras.new("dron")
cam_dat.sensor_fit = "HORIZONTAL"
cam_dat.sensor_width = float(CAM.get("sensor_mm", 36.0))
cam_dat.lens = float(CAM["focal_mm"])
cam_dat.clip_start = 0.5
cam_dat.clip_end = float(CAM.get("clip_end", 20000.0))
cam = bpy.data.objects.new("camara_dron", cam_dat)
esc.collection.objects.link(cam)

R = Matrix([[float(v) for v in fila] for fila in CAM["R_c_w"]])   # OpenCV c->w
R = R @ Matrix.Diagonal((1.0, -1.0, -1.0))                        # -> Blender
M = R.to_4x4()
M.translation = Vector(CAM["posicion_blender"])
cam.matrix_world = M
esc.camera = cam

# ══════════════════════════════════════════════════════════════════════════
# 4. SOL  (azimut y elevacion del instante de la detonacion)
# ══════════════════════════════════════════════════════════════════════════
def crear_sol(nombre, vec_sol, sombra):
    dat = bpy.data.lights.new(nombre, type="SUN")
    dat.energy = ENERGIA_SOL
    dat.angle = math.radians(0.53)
    dat.use_shadow = sombra
    if hasattr(dat, "cycles") and hasattr(dat.cycles, "cast_shadow"):
        dat.cycles.cast_shadow = sombra
    for attr, val in (("shadow_soft_size", 2.0), ("use_shadow_jitter", True)):
        if hasattr(dat, attr):
            try:
                setattr(dat, attr, val)
            except (TypeError, ValueError):
                pass
    obj = bpy.data.objects.new(nombre, dat)
    esc.collection.objects.link(obj)
    # el sol de Blender emite a lo largo de su -Z: se apunta en -vec_sol,
    # que es la direccion en que viaja la luz y en que corre la sombra
    obj.rotation_euler = (-vec_sol).to_track_quat("-Z", "Y").to_euler()
    obj.location = vec_sol * 500.0
    return obj


if "sol" in CAM:
    s = CAM["sol"]
    vec_sol = Vector(s["vec_sol"])            # DESDE el suelo HACIA el sol
    _hay_linking = "light_linking" in bpy.types.Object.bl_rna.properties
    if DOS_SOLES and _hay_linking:
        sol_ter = crear_sol("sol_terreno", vec_sol, SOMBRAS_SOL)
        sol_ter.light_linking.receiver_collection = col_ter
        sol_gas = crear_sol("sol_gas", vec_sol, False)
        sol_gas.light_linking.receiver_collection = col_gas
        print("[sol] dos soles con light linking: 'sol_terreno' ilumina el "
              "terreno y recibe la\n      sombra del gas; 'sol_gas' ilumina "
              "el gas sin sombrearlo por dentro")
        if not esc.render.engine == "CYCLES":
            print("[aviso] el light linking esta pensado para Cycles; en "
                  "EEVEE puede no respetarse")
    else:
        if DOS_SOLES:
            print("[aviso] esta version de Blender no tiene light linking "
                  "(requiere 4.0+): se usa un solo sol")
        crear_sol("sol", vec_sol, SOMBRAS_SOL)
    print(f"[sol] azimut {s['azimut_deg']:.2f} deg  "
          f"elevacion {s['elevacion_deg']:.2f} deg")

# ══════════════════════════════════════════════════════════════════════════
# 5. ANIMACION: cada hull visible en su tramo
# ══════════════════════════════════════════════════════════════════════════
keys = CAM.get("keyframes")
if keys is None or len(keys) != len(hulls):
    print(f"[aviso] el JSON trae {0 if keys is None else len(keys)} keyframes y "
          f"hay {len(hulls)} hulls. Se anima uno por frame de Blender.")
    keys = list(range(len(hulls)))
    f0, fps_video = 0, esc.render.fps
else:
    f0 = keys[0]

# frame de video -> frame de Blender, en tiempo real
a_blender = lambda f: int(round((f - f0) * esc.render.fps / fps_video)) + 1

tramos = []
for i, f in enumerate(keys):
    b0 = a_blender(f)
    b1 = (a_blender(keys[i + 1]) - 1) if i + 1 < len(keys) else b0 + int(esc.render.fps // 4)
    tramos.append((max(b0, 1), max(b1, b0)))

esc.frame_start = 1
esc.frame_end = max(t[1] for t in tramos)

def _fcurves(accion):
    """Itera las F-curves de una accion en cualquier version de Blender.

    Hasta 4.3 estaban en `accion.fcurves`. En 4.4 llegaron las acciones con
    capas y slots y ese atributo desaparecio: ahora cuelgan de
    layers -> strips -> channelbags -> fcurves.
    """
    fc = getattr(accion, "fcurves", None)
    if fc is not None:
        return list(fc)
    out = []
    for capa in getattr(accion, "layers", []):
        for strip in getattr(capa, "strips", []):
            bolsas = getattr(strip, "channelbags", None)
            if bolsas is None:
                continue
            for b in bolsas:
                out.extend(getattr(b, "fcurves", []))
    return out


# La visibilidad tiene que saltar, no interpolarse. En vez de corregir las
# curvas despues -que es lo que rompio en 4.4- se pide CONSTANT de entrada,
# via la preferencia que gobierna los keyframes nuevos.
_prefs = bpy.context.preferences.edit
_interp_previa = _prefs.keyframe_new_interpolation_type
_prefs.keyframe_new_interpolation_type = "CONSTANT"
try:
    for o, (b0, b1) in zip(hulls, tramos):
        o.hide_viewport = True
        o.hide_render = True
        for prop in ("hide_viewport", "hide_render"):
            if b0 > 1:
                setattr(o, prop, True)
                o.keyframe_insert(prop, frame=1)
            setattr(o, prop, False)
            o.keyframe_insert(prop, frame=b0)
            setattr(o, prop, True)
            o.keyframe_insert(prop, frame=b1 + 1)
finally:
    _prefs.keyframe_new_interpolation_type = _interp_previa

# ── color del gas: un keyframe por tramo, en el material compartido ───────
# Solo hay un hull visible a la vez, asi que animar el material comun basta.
if _cs:
    _nodo = mat.node_tree.nodes["volumen_gas"]
    _entradas = [_nodo.inputs["Color"]]
    if "Emission Color" in _nodo.inputs:
        _entradas.append(_nodo.inputs["Emission Color"])
    _prefs.keyframe_new_interpolation_type = "CONSTANT"
    try:
        _n_col = 0
        for f, (b0, b1) in zip(keys, tramos):
            c = _cs.get(int(f))
            if c is None:
                continue
            lin = (srgb_a_lineal(c[0]), srgb_a_lineal(c[1]),
                   srgb_a_lineal(c[2]), 1.0)
            for _e in _entradas:
                _e.default_value = lin
                _e.keyframe_insert("default_value", frame=b0)
            _n_col += 1
    finally:
        _prefs.keyframe_new_interpolation_type = _interp_previa
    print(f"[color] {_n_col} keyframes de color aplicados al volumen")

# Cinturon y tirantes: si la version lo permite, se fuerza tambien en las
# curvas ya creadas. Si no se puede, no importa: ya se insertaron CONSTANT.
_n_fix = 0
for o in hulls:
    ad = o.animation_data
    if not (ad and ad.action):
        continue
    try:
        for fc in _fcurves(ad.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
                _n_fix += 1
    except Exception as _e:
        print(f"[aviso] no se pudieron recorrer las F-curves ({_e}). "
              f"No es grave: los keyframes ya se insertaron en CONSTANT.")
        break

# ══════════════════════════════════════════════════════════════════════════
# 6. TEXTO CON LA ALTURA  (opcional)
# ══════════════════════════════════════════════════════════════════════════
if MOSTRAR_ALTURA and "altura_m" in CAM:
    alturas = {int(k): float(v) for k, v in CAM["altura_m"].items()}
    txt = bpy.data.curves.new("altura", type="FONT")
    txt.align_x = "LEFT"
    txt.align_y = "TOP"
    txt.size = 1.0
    obj_txt = bpy.data.objects.new("altura_txt", txt)
    esc.collection.objects.link(obj_txt)

    # El tamano se deriva del encuadre, no de un numero suelto: a distancia D
    # delante de la camara el cuadro mide 2*hw x 2*hh, asi que pedir "4.5% del
    # alto" da el mismo resultado con cualquier focal. Antes era size=8 por
    # scale=0.02, que ocupaba una quinta parte del cuadro.
    D = 1.0
    hw = D * math.tan(math.radians(float(CAM["fov_h_deg"])) / 2.0)
    hh = hw * float(CAM["alto_px"]) / float(CAM["ancho_px"])
    alto_txt = ALTO_TEXTO * (2.0 * hh)
    obj_txt.parent = cam
    obj_txt.scale = (alto_txt,) * 3
    obj_txt.location = (-hw + MARGEN_TEXTO * 2.0 * hw,
                        hh - MARGEN_TEXTO * 2.0 * hh, -D)
    obj_txt.rotation_euler = (0.0, 0.0, 0.0)

    def _txt_handler(escena):
        b = escena.frame_current
        act = None
        for f, (b0, b1) in zip(keys, tramos):
            if b0 <= b <= b1:
                act = f
                break
        h = alturas.get(int(act)) if act is not None else None
        obj_txt.data.body = f"t {(b - 1) / escena.render.fps:5.2f} s   h {h:5.1f} m" \
            if h is not None else ""

    # quitar solo versiones previas de ESTE handler, no los de otros addons
    for _h in list(bpy.app.handlers.frame_change_pre):
        if getattr(_h, "__name__", "") == "_txt_handler":
            bpy.app.handlers.frame_change_pre.remove(_h)
    bpy.app.handlers.frame_change_pre.append(_txt_handler)
    _txt_handler(esc)
    print("[texto] la altura se actualiza con un handler de frame_change. "
          "Para render en linea de comandos, ejecuta este script con --python.")

# ══════════════════════════════════════════════════════════════════════════
esc.frame_set(esc.frame_start)
print(f"\n  escena lista: frames {esc.frame_start}-{esc.frame_end} a "
      f"{esc.render.fps} fps ({(esc.frame_end)/esc.render.fps:.1f} s)")
print(f"  origen UTM: E {CAM['origen_utm'][0]:.2f} N {CAM['origen_utm'][1]:.2f} "
      f"z {CAM['origen_utm'][2]:.2f}")
print("  +X = Este  +Y = Norte  +Z = arriba")
print("  Numpad 0 para mirar por la camara del dron.")
print("\n  Si el penacho sigue oscuro, en este orden:")
print("    1. baja TAU_GAS  (1.2 -> 0.8): menos denso, entra mas luz")
print("    2. sube EMISION_GAS (0.35 -> 0.8): piso de brillo, no puede salir negro")
print("    3. sube LUZ_AMBIENTE (1.2 -> 2.0): mas luz de cielo")
print("  Si se ve pixelado al reproducir, MOTOR ya deberia ser 'EEVEE'. "
      "Con Cycles\n  el viewport es progresivo y eso no se arregla con ajustes.")
print("\n  Si no se ve la SOMBRA del penacho sobre el terreno:")
print("    1. MOTOR = 'CYCLES' y mira un frame quieto. EEVEE la aproxima mal.")
print("    2. baja LUZ_AMBIENTE (0.35 -> 0.15): mas contraste sol/sombra")
print("    3. sube TAU_GAS (1.2 -> 2.0): un penacho mas denso tapa mas luz")
print("    4. EMISION_GAS alto tambien lava la escena; bajalo si sobra luz")
'''


def escribir_script_blender(dir_salidas, ruta_script=None, verbose=True):
    """Escribe `blender_animar_gas.py` apuntando ya a `dir_salidas`."""
    dir_salidas = os.path.abspath(dir_salidas)
    if ruta_script is None:
        ruta_script = os.path.join(dir_salidas, "blender_animar_gas.py")
    txt = _PLANTILLA_BLENDER.replace("@@RUTA_SALIDAS@@", dir_salidas)
    with open(ruta_script, "w", encoding="utf-8") as fh:
        fh.write(txt)
    if verbose:
        print(f"  script de Blender: {F0.ruta_corta(ruta_script)}")
    return ruta_script
