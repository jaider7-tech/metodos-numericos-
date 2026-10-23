"""
Problemas de valor inicial  dy/dx = f(x, y),  y(x0) = y0.

Todos los metodos son "un paso":  y_{n+1} = y_n + h · (pendiente promedio)
Solo cambia como se calcula esa pendiente, asi que comparten un motor.

    Euler                 k1
    Cauchy-Euler (Heun)   (k1 + k2)/2,   k2 = f(x+h, y + h k1)      [predictor-corrector]
    Punto medio (RK2)     k2 = f(x+h/2, y + h/2 k1)
    Runge-Kutta 4         (k1 + 2k2 + 2k3 + k4)/6
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sympy as sp


class ErrorDeEDO(Exception):
    pass


@dataclass
class ResultadoEDO:
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    mensaje: str = ""


def _paso_euler(f, x, y, h):
    k1 = f(x, y)
    return y + h * k1, {"k1 = f(x_n, y_n)": k1}


def _paso_heun(f, x, y, h):
    k1 = f(x, y)
    pred = y + h * k1
    k2 = f(x + h, pred)
    return y + h / 2 * (k1 + k2), {"k1 = f(x_n, y_n)": k1, "y* (predictor)": pred,
                                   "k2 = f(x_n+h, y*)": k2}


def _paso_medio(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    return y + h * k2, {"k1": k1, "k2 = f(x+h/2, y+h/2·k1)": k2}


def _paso_rk4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), {"k1": k1, "k2": k2, "k3": k3, "k4": k4}


METODOS = {
    "Euler": _paso_euler,
    "Cauchy-Euler (Heun / Euler mejorado)": _paso_heun,
    "Punto medio (RK2)": _paso_medio,
    "Runge-Kutta 4 (RK4)": _paso_rk4,
}


def resolver(f, x0, y0, h, x_final, metodo="Runge-Kutta 4 (RK4)", exacta=None):
    if h == 0:
        raise ErrorDeEDO("El paso h no puede ser 0.")
    pasos = (x_final - x0) / h
    if pasos <= 0:
        raise ErrorDeEDO("x final debe estar en la direccion del paso h (x final > x0 si h > 0).")
    n = int(round(pasos))
    if abs(pasos - n) > 1e-9:
        raise ErrorDeEDO(
            f"(x_final − x0)/h = {pasos:g} no es entero. Ajusta h o x final "
            f"(por ejemplo h = {(x_final - x0) / max(1, round(pasos)):g})."
        )
    if n > 100000:
        raise ErrorDeEDO("Demasiados pasos (mas de 100 000). Aumenta h.")

    paso = METODOS[metodo]
    x, y = float(x0), float(y0)
    xs, ys = [x], [y]
    fila0 = {"n": 0, "x_n": x, "y_n": y}
    if exacta:
        fila0["y exacta"] = exacta(x)
        fila0["Error abs"] = abs(fila0["y exacta"] - y)
    filas = [fila0]
    mensaje = f"{n} pasos de tamaño h = {h:g}."
    for k in range(1, n + 1):
        y_nuevo, ks = paso(f, x, y, h)
        x = x0 + k * h
        y = y_nuevo
        if not np.isfinite(y):
            mensaje = f"La solucion se dispara en el paso {k} (x = {x:g}). Prueba un h mas pequeño."
            break
        fila = {"n": k, **{c: v for c, v in ks.items()}, "x_n": x, "y_n": y}
        if exacta:
            ye = exacta(x)
            fila["y exacta"] = ye
            fila["Error abs"] = abs(ye - y)
        filas.append(fila)
        xs.append(x)
        ys.append(y)

    # orden: n, x_n, (k...), y_n, exacta, error
    tabla = pd.DataFrame(filas)
    orden = ["n", "x_n"] + [c for c in tabla.columns if c not in ("n", "x_n", "y_n", "y exacta", "Error abs")] \
            + ["y_n"] + [c for c in ("y exacta", "Error abs") if c in tabla.columns]
    return ResultadoEDO(xs=xs, ys=ys, tabla=tabla[orden], mensaje=mensaje)


def solucion_exacta(expr_f, x0, y0):
    """Intenta y(x) exacta con dsolve. Devuelve (expr, funcion) o (None, None)."""
    x = sp.Symbol("x")
    Yf = sp.Function("Y")
    y = sp.Symbol("y")
    try:
        ecuacion = sp.Eq(Yf(x).diff(x), expr_f.subs(y, Yf(x)))
        sol = sp.dsolve(ecuacion, Yf(x), ics={Yf(x0): y0})
        if isinstance(sol, list):
            sol = sol[0]
        expr = sp.simplify(sol.rhs)
        if expr.has(sp.Integral) or expr.free_symbols - {x}:
            return None, None
        g = sp.lambdify(x, expr, "numpy")
        return expr, (lambda t: float(np.real(g(t))))
    except Exception:
        return None, None
