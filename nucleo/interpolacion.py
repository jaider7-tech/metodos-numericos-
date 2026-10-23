"""
Interpolacion polinomial: Lagrange, Newton (diferencias divididas), Neville y Hermite.

Los polinomios se construyen con fracciones exactas (SymPy) para que P(x) salga
limpio (por ejemplo -3/8 x^2 + 3/4 x + 3) y ademas se muestra en decimales.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sympy as sp

X = sp.symbols("x")


class ErrorDeInterpolacion(Exception):
    pass


@dataclass
class ResultadoInterp:
    polinomio: sp.Expr | None = None          # expandido, exacto
    forma: sp.Expr | None = None              # forma del metodo (bases / anidada)
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    bases: list = field(default_factory=list)  # Lagrange: [(nombre, expr)]
    valor: float | None = None                 # P(X) si se pidio
    extra: dict = field(default_factory=dict)

    def evaluar(self, xs):
        f = sp.lambdify(X, self.polinomio, "numpy")
        return np.asarray(f(np.asarray(xs, dtype=float)), dtype=float) * np.ones_like(np.asarray(xs, dtype=float))


def _exacto(v):
    return sp.nsimplify(float(v), rational=True, tolerance=1e-12)


def _validar(xs, ys):
    if len(xs) != len(ys):
        raise ErrorDeInterpolacion("Debe haber la misma cantidad de valores x que de valores y.")
    if len(xs) < 2:
        raise ErrorDeInterpolacion("Se necesitan al menos dos puntos.")
    if len(set(xs)) != len(xs):
        raise ErrorDeInterpolacion("Hay valores de x repetidos; en Lagrange/Newton/Neville deben ser distintos.")


def decimal(expr, digitos=6):
    return sp.N(expr, digitos)


# ------------------------------------------------------------- Lagrange

def lagrange(xs, ys, X0=None):
    _validar(xs, ys)
    xe = [_exacto(v) for v in xs]
    ye = [_exacto(v) for v in ys]
    n = len(xs)
    bases, P, filas = [], 0, []
    for i in range(n):
        L = 1
        for j in range(n):
            if j != i:
                L *= (X - xe[j]) / (xe[i] - xe[j])
        L = sp.expand(L)
        bases.append((f"L{i}(x)", L))
        P += ye[i] * L
        fila = {"i": i, "x_i": xs[i], "y_i": ys[i], "L_i(x)": str(L)}
        if X0 is not None:
            fila[f"L_i({X0:g})"] = float(L.subs(X, X0))
            fila[f"y_i·L_i({X0:g})"] = float(ye[i] * L.subs(X, X0))
        filas.append(fila)
    P = sp.expand(P)
    forma = sp.Add(*[sp.Mul(ye[i], sp.Symbol(f"L{i}(x)")) for i in range(n)], evaluate=False)
    valor = float(P.subs(X, X0)) if X0 is not None else None
    return ResultadoInterp(polinomio=P, forma=forma, tabla=pd.DataFrame(filas), bases=bases, valor=valor)


# ------------------------------------------------ Newton (dif. divididas)

def newton(xs, ys, X0=None):
    _validar(xs, ys)
    n = len(xs)
    xe = [_exacto(v) for v in xs]
    T = [[None] * n for _ in range(n)]
    for i in range(n):
        T[i][0] = _exacto(ys[i])
    for j in range(1, n):
        for i in range(n - j):
            T[i][j] = (T[i + 1][j - 1] - T[i][j - 1]) / (xe[i + j] - xe[i])

    columnas = ["x_i", "f[x_i]"] + [f"Orden {j}" for j in range(1, n)]
    filas = []
    for i in range(n):
        filas.append([xs[i]] + [float(T[i][j]) if T[i][j] is not None else None for j in range(n)])
    tabla = pd.DataFrame(filas, columns=columnas)

    P, producto, terminos = 0, 1, []
    for j in range(n):
        if j > 0:
            producto *= (X - xe[j - 1])
        P += T[0][j] * producto
        terminos.append(sp.Mul(T[0][j], producto, evaluate=False) if j else T[0][0])
    P = sp.expand(P)
    forma = sp.Add(*terminos, evaluate=False)
    coef = [float(T[0][j]) for j in range(n)]
    valor = float(P.subs(X, X0)) if X0 is not None else None
    return ResultadoInterp(polinomio=P, forma=forma, tabla=tabla, valor=valor,
                           extra={"coeficientes": coef})


# ------------------------------------------------------------- Neville

def neville(xs, ys, X0):
    _validar(xs, ys)
    if X0 is None:
        raise ErrorDeInterpolacion("Neville necesita el valor X donde quieres interpolar.")
    n = len(xs)
    x = np.array(xs, dtype=float)
    P = np.full((n, n), np.nan)
    P[:, 0] = ys
    for j in range(1, n):
        for i in range(n - j):
            P[i, j] = ((X0 - x[i]) * P[i + 1, j - 1] - (X0 - x[i + j]) * P[i, j - 1]) / (x[i + j] - x[i])
    columnas = ["x_i"] + [f"P(orden {j})" for j in range(n)]
    tabla = pd.DataFrame(np.column_stack((x, P)), columns=columnas)
    # etiquetas tipo P_{i,i+1,...}
    etiquetas = [[("P" + "".join(str(k) for k in range(i, i + j + 1))) if i + j < n else ""
                  for j in range(n)] for i in range(n)]
    poli = lagrange(xs, ys).polinomio                      # mismo polinomio, para graficar
    return ResultadoInterp(polinomio=poli, tabla=tabla, valor=float(P[0, n - 1]),
                           extra={"etiquetas": etiquetas})


# ------------------------------------------------------------- Hermite

def hermite(datos, X0=None):
    """
    datos: lista de filas [x, f(x), f'(x), f''(x), ...]  (minimo x y f(x)).
    Cada x se repite tantas veces como valores tenga (f, f', f''...).
    Usa diferencias divididas generalizadas:
        si z_i = z_{i+j}:  f[z_i..z_{i+j}] = f^(j)(z_i) / j!
    """
    if len(datos) < 1:
        raise ErrorDeInterpolacion("Escribe al menos un punto.")
    xs_vistos = [fila[0] for fila in datos]
    if len(set(xs_vistos)) != len(xs_vistos):
        raise ErrorDeInterpolacion("Cada x debe aparecer en una sola linea (pon sus derivadas en la misma linea).")
    derivadas = {}
    z = []
    for fila in datos:
        if len(fila) < 2:
            raise ErrorDeInterpolacion("Cada linea necesita al menos x y f(x).")
        xv = _exacto(fila[0])
        derivadas[xv] = [_exacto(v) for v in fila[1:]]
        z += [xv] * (len(fila) - 1)
    m = len(z)
    if m < 2:
        raise ErrorDeInterpolacion("Se necesitan al menos dos condiciones.")

    T = [[None] * m for _ in range(m)]
    for i in range(m):
        T[i][0] = derivadas[z[i]][0]
    for j in range(1, m):
        for i in range(m - j):
            if z[i] == z[i + j]:
                T[i][j] = derivadas[z[i]][j] / sp.factorial(j)
            else:
                T[i][j] = (T[i + 1][j - 1] - T[i][j - 1]) / (z[i + j] - z[i])

    columnas = ["z_i", "f[z_i]"] + [f"Orden {j}" for j in range(1, m)]
    filas = [[float(z[i])] + [float(T[i][j]) if T[i][j] is not None else None for j in range(m)]
             for i in range(m)]
    tabla = pd.DataFrame(filas, columns=columnas)

    H, producto, terminos = 0, 1, []
    for j in range(m):
        if j > 0:
            producto *= (X - z[j - 1])
        H += T[0][j] * producto
        terminos.append(sp.Mul(T[0][j], producto, evaluate=False) if j else T[0][0])
    H = sp.expand(H)
    dH = sp.expand(sp.diff(H, X))
    valor = float(H.subs(X, X0)) if X0 is not None else None

    verificacion = []
    for xv, vals in derivadas.items():
        for k, v in enumerate(vals):
            obtenido = sp.diff(H, X, k).subs(X, xv)
            verificacion.append({"Condicion": f"H{'′' * k}({float(xv):g})",
                                 "Pedido": float(v), "Obtenido": float(obtenido)})
    return ResultadoInterp(polinomio=H, forma=sp.Add(*terminos, evaluate=False), tabla=tabla,
                           valor=valor,
                           extra={"derivada": dH, "verificacion": pd.DataFrame(verificacion),
                                  "nodos": [(float(k), [float(v) for v in vals]) for k, vals in derivadas.items()]})
