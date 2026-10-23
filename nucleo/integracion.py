"""
Integracion numerica de f(x) en [a, b].

Newton-Cotes compuestas (nodos equiespaciados, h = (b-a)/n):
    - Trapecio            (cualquier n)
    - Simpson 1/3         (n par)
    - Simpson 3/8         (n multiplo de 3)
    - Boole               (n multiplo de 4)
Cuadratura de Gauss-Legendre con n puntos (n = 1 ... 20):
    x = (a+b)/2 + (b-a)/2 · t,   ∫ ≈ (b-a)/2 · Σ w_i f(x_i)
"""

from dataclasses import dataclass, field

import mpmath
import numpy as np
import pandas as pd


class ErrorDeIntegracion(Exception):
    pass


@dataclass
class ResultadoIntegral:
    valor: float = 0.0
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    nodos_x: list = field(default_factory=list)
    nodos_y: list = field(default_factory=list)
    formula: str = ""
    extra: dict = field(default_factory=dict)


# regla : (divisor de n, coeficiente C (h*C), patron de pesos interiores)
_REGLAS = {
    "Trapecio": (1, 1 / 2, None),
    "Simpson 1/3": (2, 1 / 3, None),
    "Simpson 3/8": (3, 3 / 8, None),
    "Boole": (4, 2 / 45, None),
}


def _pesos(regla, n):
    if regla == "Trapecio":
        w = [2] * (n + 1)
        w[0] = w[-1] = 1
    elif regla == "Simpson 1/3":
        w = [4 if i % 2 else 2 for i in range(n + 1)]
        w[0] = w[-1] = 1
    elif regla == "Simpson 3/8":
        w = [2 if i % 3 == 0 else 3 for i in range(n + 1)]
        w[0] = w[-1] = 1
    elif regla == "Boole":
        patron = {0: 14, 1: 32, 2: 12, 3: 32}
        w = [patron[i % 4] for i in range(n + 1)]
        w[0] = w[-1] = 7
    else:
        raise ErrorDeIntegracion(f"Regla desconocida: {regla}")
    return w


def newton_cotes(f, a, b, n, regla="Simpson 1/3"):
    n = int(n)
    divisor, C, _ = _REGLAS[regla]
    if n < 1:
        raise ErrorDeIntegracion("n debe ser al menos 1.")
    if n % divisor:
        raise ErrorDeIntegracion(
            f"{regla} necesita n multiplo de {divisor} (pusiste n = {n}). "
            f"Prueba n = {n + divisor - n % divisor}."
        )
    if a == b:
        raise ErrorDeIntegracion("Los limites a y b no pueden ser iguales.")
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    ys = [f(x) for x in xs]
    if not all(np.isfinite(ys)):
        malos = [f"{x:g}" for x, y in zip(xs, ys) if not np.isfinite(y)]
        raise ErrorDeIntegracion(
            "La funcion no esta definida (o no es real) en x = " + ", ".join(malos) + "."
        )
    w = _pesos(regla, n)
    suma = sum(wi * yi for wi, yi in zip(w, ys))
    valor = C * h * suma
    tabla = pd.DataFrame({"i": range(n + 1), "x_i": xs, "f(x_i)": ys, "Peso": w,
                          "Peso·f(x_i)": [wi * yi for wi, yi in zip(w, ys)]})
    coef = {"Trapecio": "h/2", "Simpson 1/3": "h/3", "Simpson 3/8": "3h/8", "Boole": "2h/45"}[regla]
    formula = f"I ≈ {coef} · Σ peso·f(x_i) = {coef} · {suma:.10g},   h = (b−a)/n = {h:.10g}"
    return ResultadoIntegral(valor=valor, tabla=tabla, nodos_x=xs, nodos_y=ys, formula=formula,
                             extra={"h": h, "suma": suma})


def nodos_gauss(n):
    n = int(n)
    if not 1 <= n <= 20:
        raise ErrorDeIntegracion("Usa entre 1 y 20 puntos de Gauss.")
    t, w = np.polynomial.legendre.leggauss(n)
    return t, w


def gauss_legendre(f, a, b, n=2):
    t, w = nodos_gauss(n)
    xs = (a + b) / 2 + (b - a) / 2 * t
    ys = [f(x) for x in xs]
    if not all(np.isfinite(ys)):
        raise ErrorDeIntegracion("La funcion no esta definida en algun nodo de Gauss.")
    prod = [wi * yi for wi, yi in zip(w, ys)]
    valor = (b - a) / 2 * sum(prod)
    tabla = pd.DataFrame({"i": range(1, n + 1), "t_i": t, "w_i": w, "x_i": xs,
                          "f(x_i)": ys, "w_i·f(x_i)": prod})
    formula = (f"x = {(a + b) / 2:g} + {(b - a) / 2:g}·t   →   "
               f"I ≈ {(b - a) / 2:g} · Σ w_i f(x_i) = {(b - a) / 2:g} · {sum(prod):.10g}")
    return ResultadoIntegral(valor=valor, tabla=tabla, nodos_x=list(xs), nodos_y=ys, formula=formula,
                             extra={"exacto_hasta_grado": 2 * n - 1})


def integral_referencia(expr_sympy, a, b, simbolo):
    """Valor 'exacto' de referencia con mpmath (precision alta, no se cuelga)."""
    import sympy as sp
    try:
        g = sp.lambdify(simbolo, expr_sympy, modules="mpmath")
        mpmath.mp.dps = 30
        v = mpmath.quad(lambda t: g(t), [a, b])
        if isinstance(v, mpmath.mpc):
            if abs(v.imag) > 1e-10:
                return None
            v = v.real
        return float(v)
    except Exception:
        return None
    finally:
        mpmath.mp.dps = 15


def integral_simbolica(expr_sympy, a, b, simbolo):
    """Intenta la integral exacta en forma cerrada; None si SymPy no la encuentra rapido."""
    import sympy as sp
    try:
        F = sp.integrate(expr_sympy, simbolo)
        if F.has(sp.Integral):
            return None, None
        return F, sp.simplify(F.subs(simbolo, b) - F.subs(simbolo, a))
    except Exception:
        return None, None


def newton_cotes_tabla(xs, ys, regla="Simpson 1/3"):
    """Misma regla, pero con datos de una tabla (x equiespaciados) en vez de f(x)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ErrorDeIntegracion("Escribe al menos dos puntos (x, y).")
    orden = np.argsort(xs)
    xs = [float(xs[i]) for i in orden]
    ys = [float(ys[i]) for i in orden]
    n = len(xs) - 1
    h = (xs[-1] - xs[0]) / n
    if any(abs((xs[i + 1] - xs[i]) - h) > 1e-9 * max(1, abs(h)) for i in range(n)):
        raise ErrorDeIntegracion("Los x de la tabla deben estar igualmente espaciados.")
    datos = dict(zip(xs, ys))
    return newton_cotes(lambda t: datos[min(datos, key=lambda k: abs(k - t))], xs[0], xs[-1], n, regla)
