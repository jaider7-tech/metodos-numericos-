"""
Metodos para encontrar raices de f(x) = 0.

IDEA CENTRAL DE LA OPTIMIZACION
-------------------------------
Biseccion, punto fijo, regula falsi, Newton-Raphson y la secante parecen
cinco algoritmos distintos, pero todos hacen exactamente lo mismo:

    1. parten de un estado inicial
    2. aplican una regla para obtener una nueva aproximacion
    3. miden el error
    4. repiten hasta que el error sea menor que la tolerancia

Lo unico que cambia entre ellos es el paso 2.  Por eso aqui hay UN SOLO
motor de iteracion (`_ejecutar`) y cada metodo aporta unicamente su regla,
que ocupa entre cuatro y ocho lineas.
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
    historial: list = field(default_factory=list)   # aproximaciones sucesivas


# --------------------------------------------------------------- el motor

def _ejecutar(paso, estado, tol, max_iter):
    """
    Motor de iteracion compartido por los cinco metodos.

    `paso` recibe el estado actual y devuelve:
        (nueva_aproximacion, nuevo_estado, fila_de_la_tabla)
    donde la fila incluye la columna "Error".
    """
    filas, historial = [], []
    raiz, convergio, mensaje = None, False, ""

    for k in range(1, int(max_iter) + 1):
        try:
            raiz, estado, fila = paso(estado)
        except ZeroDivisionError:
            mensaje = f"Division por cero en la iteracion {k}: el metodo se detiene."
            break
        except ErrorDeMetodo as error:
            mensaje = f"Iteracion {k}: {error}"
            break

        if not np.isfinite(raiz) or abs(raiz) > DESBORDE:
            mensaje = f"El metodo diverge (iteracion {k}). Prueba otro valor inicial."
            raiz = None
            break

        filas.append({"Iter": k, **fila})
        historial.append(raiz)

        if fila["Error"] <= tol:
            convergio = True
            mensaje = f"Converge en {k} iteraciones con error {fila['Error']:.2e}."
            break
    else:
        mensaje = (
            f"Se alcanzo el maximo de {int(max_iter)} iteraciones sin llegar "
            f"a la tolerancia pedida."
        )

    return Resultado(
        raiz=raiz,
        tabla=pd.DataFrame(filas),
        convergio=convergio,
        mensaje=mensaje,
        iteraciones=len(filas),
        historial=historial,
    )


# ------------------------------------------------------------- los metodos

def biseccion(f, a, b, tol=1e-6, max_iter=100):
    """Parte el intervalo por la mitad conservando el cambio de signo."""
    fa, fb = f(a), f(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        raise ErrorDeMetodo("La funcion no esta definida en algun extremo del intervalo.")
    if fa * fb > 0:
        raise ErrorDeMetodo(
            f"No hay cambio de signo: f({a:g}) = {fa:g} y f({b:g}) = {fb:g} "
            "tienen el mismo signo. Elige otro intervalo."
        )

    def paso(estado):
        a, b = estado
        c = (a + b) / 2
        fa, fc = f(a), f(c)
        fila = {
            "a": a, "b": b, "c = (a+b)/2": c,
            "f(a)": fa, "f(c)": fc, "Error": (b - a) / 2,
        }
        nuevo = (a, c) if fa * fc < 0 else (c, b)
        return c, nuevo, fila

    return _ejecutar(paso, (a, b), tol, max_iter)


def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    """Como la biseccion, pero corta con la recta que une (a,f(a)) y (b,f(b))."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ErrorDeMetodo(
            f"No hay cambio de signo: f({a:g}) = {fa:g} y f({b:g}) = {fb:g}. "
            "Elige otro intervalo."
        )

    def paso(estado):
        a, b, anterior = estado
        fa, fb = f(a), f(b)
        if abs(fb - fa) < CERO:
            raise ErrorDeMetodo("f(a) y f(b) son practicamente iguales.")
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(c - anterior) if anterior is not None else abs(b - a)
        fila = {"a": a, "b": b, "c": c, "f(a)": fa, "f(c)": fc, "Error": error}
        nuevo = (a, c, c) if fa * fc < 0 else (c, b, c)
        return c, nuevo, fila

    return _ejecutar(paso, (a, b, None), tol, max_iter)


def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    """Itera x = g(x) hasta que el valor deje de moverse."""

    def paso(x):
        siguiente = g(x)
        fila = {"x_k": x, "g(x_k)": siguiente, "Error": abs(siguiente - x)}
        return siguiente, siguiente, fila

    return _ejecutar(paso, x0, tol, max_iter)


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """Avanza por la recta tangente. La derivada la calcula SymPy sola."""

    def paso(x):
        fx, dfx = f(x), df(x)
        if abs(dfx) < CERO:
            raise ErrorDeMetodo(
                f"La derivada vale casi cero en x = {x:g}: la tangente es horizontal."
            )
        siguiente = x - fx / dfx
        fila = {
            "x_k": x, "f(x_k)": fx, "f'(x_k)": dfx,
            "x_(k+1)": siguiente, "Error": abs(siguiente - x),
        }
        return siguiente, siguiente, fila

    return _ejecutar(paso, x0, tol, max_iter)


def secante(f, x0, x1, tol=1e-6, max_iter=100):
    """Newton sin derivada: usa una recta por los dos ultimos puntos."""

    def paso(estado):
        x0, x1 = estado
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < CERO:
            raise ErrorDeMetodo("f(x0) y f(x1) son practicamente iguales.")
        siguiente = x1 - f1 * (x1 - x0) / (f1 - f0)
        fila = {
            "x_(k-1)": x0, "x_k": x1, "f(x_(k-1))": f0, "f(x_k)": f1,
            "x_(k+1)": siguiente, "Error": abs(siguiente - x1),
        }
        return siguiente, (x1, siguiente), fila

    return _ejecutar(paso, (x0, x1), tol, max_iter)
