"""
Metodos para encontrar raices de f(x) = 0.

IDEA CENTRAL
------------
Biseccion, punto fijo, regula falsi, Newton-Raphson y secante hacen lo mismo:
parten de un estado, aplican una regla, miden el error y repiten.  Solo cambia
la regla, asi que hay UN motor (`_ejecutar`) y cada metodo aporta su regla.

Criterios de parada (iguales para los cinco):
    - "tolerancia":  se detiene cuando el error elegido <= tol
    - "iteraciones": hace exactamente N iteraciones (como en los ejercicios
                     de clase: "usa 4 iteraciones")
Tipo de error: "absoluto" |x_k+1 - x_k|  o  "relativo %" |(x_k+1 - x_k)/x_k+1|*100
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CERO = 1e-14
DESBORDE = 1e14


class ErrorDeMetodo(Exception):
    """El metodo no se puede aplicar con los datos entregados."""


@dataclass
class Resultado:
    raiz: float | None = None
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergio: bool = False
    mensaje: str = ""
    iteraciones: int = 0
    historial: list = field(default_factory=list)


def _errores(nuevo, anterior):
    abs_ = abs(nuevo - anterior)
    rel = abs_ / abs(nuevo) * 100 if abs(nuevo) > CERO else float("inf")
    return abs_, rel


def _ejecutar(paso, estado, tol, max_iter, criterio="tolerancia", tipo_error="absoluto"):
    """
    `paso(estado)` devuelve (nueva_aprox, nuevo_estado, fila, anterior)
    `anterior` es la aproximacion previa usada para medir el error
    (o un error ya calculado en fila["Error abs"] si el metodo lo define distinto).
    """
    filas, historial = [], []
    raiz, convergio, mensaje = None, False, ""
    columna = "Error abs" if tipo_error == "absoluto" else "Error rel %"

    for k in range(1, int(max_iter) + 1):
        try:
            raiz, estado, fila = paso(estado)
        except ZeroDivisionError:
            mensaje = f"Division por cero en la iteracion {k}: el metodo se detiene."
            break
        except ErrorDeMetodo as error:
            mensaje = f"Iteracion {k}: {error}"
            break

        if raiz is None or not np.isfinite(raiz) or abs(raiz) > DESBORDE:
            mensaje = f"El metodo diverge o sale del dominio (iteracion {k}). Prueba otro valor inicial."
            raiz = historial[-1] if historial else None
            break

        filas.append({"Iter": k, **fila})
        historial.append(raiz)

        if criterio == "tolerancia" and fila[columna] <= tol:
            convergio = True
            mensaje = f"Converge en {k} iteraciones ({columna.lower()} = {fila[columna]:.3e} ≤ {tol:g})."
            break
    else:
        if criterio == "iteraciones":
            convergio = True
            mensaje = f"Se hicieron las {int(max_iter)} iteraciones pedidas."
        else:
            mensaje = f"Se alcanzo el maximo de {int(max_iter)} iteraciones sin llegar a la tolerancia."

    return Resultado(raiz=raiz, tabla=pd.DataFrame(filas), convergio=convergio,
                     mensaje=mensaje, iteraciones=len(filas), historial=historial)


def _comprobar_intervalo(f, a, b):
    if a >= b:
        raise ErrorDeMetodo("El extremo a debe ser menor que b.")
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ErrorDeMetodo("La funcion no esta definida en algun extremo del intervalo.")
    if fa * fb > 0:
        raise ErrorDeMetodo(
            f"No hay cambio de signo: f({a:g}) = {fa:.6g} y f({b:g}) = {fb:.6g} "
            "tienen el mismo signo. Elige otro intervalo (mira la grafica)."
        )


# ------------------------------------------------------------- metodos

def biseccion(f, a, b, tol=1e-6, max_iter=100, criterio="tolerancia", tipo_error="absoluto"):
    """c = (a+b)/2; conserva la mitad con cambio de signo. Error abs = (b-a)/2."""
    _comprobar_intervalo(f, a, b)

    def paso(estado):
        a, b, anterior = estado
        c = (a + b) / 2
        fa, fb, fc = f(a), f(b), f(c)
        abs_ = (b - a) / 2
        rel = abs((c - anterior) / c) * 100 if anterior is not None and abs(c) > CERO else abs_ / max(abs(c), CERO) * 100
        fila = {"a": a, "b": b, "c=(a+b)/2": c, "f(a)": fa, "f(b)": fb, "f(c)": fc,
                "Error abs": abs_, "Error rel %": rel}
        if fc == 0:
            return c, (c, c, c), {**fila, "Error abs": 0.0, "Error rel %": 0.0}
        nuevo = (a, c, c) if fa * fc < 0 else (c, b, c)
        return c, nuevo, fila

    return _ejecutar(paso, (a, b, None), tol, max_iter, criterio, tipo_error)


def regula_falsi(f, a, b, tol=1e-6, max_iter=100, criterio="tolerancia", tipo_error="absoluto"):
    """c = (a f(b) - b f(a)) / (f(b) - f(a)): corta con la recta secante."""
    _comprobar_intervalo(f, a, b)

    def paso(estado):
        a, b, anterior = estado
        fa, fb = f(a), f(b)
        if abs(fb - fa) < CERO:
            raise ErrorDeMetodo("f(a) y f(b) son practicamente iguales.")
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        ref = anterior if anterior is not None else (a if abs(c - a) > abs(c - b) else b)
        abs_, rel = _errores(c, ref)
        fila = {"a": a, "b": b, "f(a)": fa, "f(b)": fb, "c": c, "f(c)": fc,
                "Error abs": abs_, "Error rel %": rel}
        if fc == 0:
            return c, (c, c, c), {**fila, "Error abs": 0.0, "Error rel %": 0.0}
        nuevo = (a, c, c) if fa * fc < 0 else (c, b, c)
        return c, nuevo, fila

    return _ejecutar(paso, (a, b, None), tol, max_iter, criterio, tipo_error)


def punto_fijo(g, x0, tol=1e-6, max_iter=100, criterio="tolerancia", tipo_error="absoluto",
               raiz_exacta=None):
    """x_{k+1} = g(x_k). Si se da la raiz exacta, agrega la columna |raiz - x|."""

    def paso(x):
        sig = g(x)
        abs_, rel = _errores(sig, x)
        fila = {"x_k": x, "x_k+1 = g(x_k)": sig, "Error abs": abs_, "Error rel %": rel}
        if raiz_exacta is not None and np.isfinite(sig):
            fila["|raiz exacta - x_k+1|"] = abs(raiz_exacta - sig)
        return sig, sig, fila

    return _ejecutar(paso, x0, tol, max_iter, criterio, tipo_error)


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100, criterio="tolerancia", tipo_error="absoluto"):
    """x_{k+1} = x_k - f(x_k)/f'(x_k)."""

    def paso(x):
        fx, dfx = f(x), df(x)
        if not np.isfinite(dfx) or abs(dfx) < CERO:
            raise ErrorDeMetodo(f"f'(x) ≈ 0 en x = {x:g}: la tangente es horizontal.")
        sig = x - fx / dfx
        abs_, rel = _errores(sig, x)
        fila = {"x_k": x, "f(x_k)": fx, "f'(x_k)": dfx, "x_k+1": sig,
                "Error abs": abs_, "Error rel %": rel}
        return sig, sig, fila

    return _ejecutar(paso, x0, tol, max_iter, criterio, tipo_error)


def secante(f, x0, x1, tol=1e-6, max_iter=100, criterio="tolerancia", tipo_error="absoluto"):
    """Newton sin derivada: la tangente se cambia por la recta entre dos puntos."""

    def paso(estado):
        xa, xb = estado
        fa, fb = f(xa), f(xb)
        if abs(fb - fa) < CERO:
            raise ErrorDeMetodo("f(x_k) y f(x_k-1) son practicamente iguales.")
        sig = xb - fb * (xb - xa) / (fb - fa)
        abs_, rel = _errores(sig, xb)
        fila = {"x_k-1": xa, "x_k": xb, "f(x_k-1)": fa, "f(x_k)": fb, "x_k+1": sig,
                "Error abs": abs_, "Error rel %": rel}
        return sig, (xb, sig), fila

    return _ejecutar(paso, (x0, x1), tol, max_iter, criterio, tipo_error)


def buscar_intervalos(f, desde=-10.0, hasta=10.0, pasos=2000):
    """Recorre [desde, hasta] y devuelve los subintervalos donde f cambia de signo."""
    xs = np.linspace(desde, hasta, pasos + 1)
    ys = np.array([f(x) for x in xs])
    encontrados = []
    for i in range(pasos):
        y1, y2 = ys[i], ys[i + 1]
        if np.isfinite(y1) and np.isfinite(y2) and y1 * y2 <= 0 and abs(y2 - y1) < 1e6:
            encontrados.append((round(xs[i], 6), round(xs[i + 1], 6)))
    return encontrados
