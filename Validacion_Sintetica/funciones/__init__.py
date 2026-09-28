"""Paquete del pipeline: agrupa funciones_F0..F4 y permite recargarlos sin reiniciar el kernel."""

import importlib
import os
import sys

DIR_PAQUETE = os.path.dirname(os.path.abspath(__file__))
if DIR_PAQUETE not in sys.path:
    sys.path.insert(0, DIR_PAQUETE)

import funciones_F0 as F0
import funciones_F1 as F1
import funciones_F2 as F2
import funciones_F3 as F3
import funciones_F4 as F4

MODULOS = (F0, F1, F2, F3, F4)

__all__ = ["F0", "F1", "F2", "F3", "F4", "MODULOS", "fases", "recargar"]


def recargar():
    """Relee los cinco modulos del disco, F0 primero."""
    for m in MODULOS:
        importlib.reload(m)
    return MODULOS


def fases(recargar_antes=True):
    """Devuelve (F0, F1, F2, F3, F4), recargandolos por defecto."""
    if recargar_antes:
        recargar()
    return MODULOS
