"""
Sistemas de ecuaciones lineales A x = b.

- Gauss-Jordan  (con pivoteo parcial opcional y registro de cada paso)
- Gauss-Seidel y Gauss-Jacobi (la misma funcion; cambia un booleano)
- Algoritmo de Thomas (tridiagonales)
"""

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import pandas as pd

CERO = 1e-14


class ErrorDeSistema(Exception):
    pass


@dataclass
class ResultadoSistema:
    solucion: np.ndarray | None = None
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    pasos: list = field(default_factory=list)      # [(titulo, DataFrame)]
    mensaje: str = ""
    convergio: bool = True
    extra: dict = field(default_factory=dict)


def _matriz_aumentada(M, nombres):
    return pd.DataFrame(M, columns=list(nombres) + ["| b"])


# ----------------------------------------------------------- Gauss-Jordan

def gauss_jordan(A, b, nombres, pivoteo=True):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    pasos = [("Matriz aumentada inicial", _matriz_aumentada(M.copy(), nombres))]

    for i in range(n):
        if pivoteo:
            p = i + int(np.argmax(np.abs(M[i:, i])))
        else:
            p = i
            if abs(M[i, i]) < CERO:                  # sin pivoteo: solo cambia si hay cero
                candidatos = [k for k in range(i + 1, n) if abs(M[k, i]) > CERO]
                p = candidatos[0] if candidatos else i
        if abs(M[p, i]) < CERO:
            raise ErrorDeSistema("La matriz es singular: el sistema no tiene solucion unica.")
        if p != i:
            M[[i, p]] = M[[p, i]]
            pasos.append((f"Intercambio F{i+1} ↔ F{p+1}", _matriz_aumentada(M.copy(), nombres)))

        piv = M[i, i]
        M[i] = M[i] / piv
        operaciones = [f"F{i+1} ← F{i+1} / {piv:.6g}"]
        for j in range(n):
            if j != i and abs(M[j, i]) > CERO:
                factor = M[j, i]
                M[j] = M[j] - factor * M[i]
                operaciones.append(f"F{j+1} ← F{j+1} − ({factor:.6g})·F{i+1}")
        pasos.append(("; ".join(operaciones), _matriz_aumentada(M.copy(), nombres)))

    x = M[:, -1].copy()
    tabla = pd.DataFrame({"Incognita": nombres, "Valor": x})
    return ResultadoSistema(solucion=x, tabla=tabla, pasos=pasos,
                            mensaje="Matriz reducida a la identidad: la ultima columna es la solucion.")


# ------------------------------------------------- Seidel / Jacobi

def radio_espectral(A, seidel):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    try:
        if seidel:
            T = -np.linalg.solve(D + L, U)
        else:
            T = -np.linalg.solve(D, L + U)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(max(abs(np.linalg.eigvals(T))))


def es_diagonal_dominante(A):
    d = np.abs(np.diag(A))
    resto = np.sum(np.abs(A), axis=1) - d
    return bool(np.all(d >= resto) and np.any(d > resto))


def mejor_orden(A, seidel):
    """Prueba reordenar filas para que el metodo converja (menor radio espectral)."""
    n = len(A)
    original = radio_espectral(A, seidel) if np.all(np.abs(np.diag(A)) > CERO) else float("inf")
    if n > 7:
        return list(range(n)), original, original
    mejor, rho_mejor = list(range(n)), original
    for perm in permutations(range(n)):
        B = A[list(perm)]
        if np.any(np.abs(np.diag(B)) < CERO):
            continue
        rho = radio_espectral(B, seidel)
        if rho < rho_mejor - 1e-12:
            mejor, rho_mejor = list(perm), rho
    return mejor, original, rho_mejor


def iterativo(A, b, nombres, seidel=True, x0=None, tol=1e-6, max_iter=100,
              criterio="tolerancia", tipo_error="relativo %", reordenar=True):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    extra = {}

    if reordenar:
        orden, rho0, rho1 = mejor_orden(A, seidel)
        extra.update({"rho_original": rho0, "rho_final": rho1, "orden": orden})
        if orden != list(range(n)):
            A, b = A[orden], b[orden]
            extra["reordenado"] = True
    else:
        extra["rho_final"] = radio_espectral(A, seidel)
    if np.any(np.abs(np.diag(A)) < CERO):
        raise ErrorDeSistema("Hay un cero en la diagonal; reordena las ecuaciones.")

    extra["A"], extra["b"] = A, b
    extra["despejes"] = [
        f"{nombres[i]} = (" + f"{b[i]:g}" + "".join(
            f" {'-' if A[i, j] > 0 else '+'} {abs(A[i, j]):g}{nombres[j]}"
            for j in range(n) if j != i and abs(A[i, j]) > CERO
        ) + f") / {A[i, i]:g}"
        for i in range(n)
    ]

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    filas = [{"Iter": 0, **{nombres[i]: x[i] for i in range(n)}}]
    convergio, mensaje = False, ""

    for k in range(1, int(max_iter) + 1):
        viejo = x.copy()
        nuevo = x.copy()
        for i in range(n):
            fuente = nuevo if seidel else viejo
            s = sum(A[i, j] * fuente[j] for j in range(n) if j != i)
            nuevo[i] = (b[i] - s) / A[i, i]
        x = nuevo

        abs_ = np.abs(x - viejo)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.abs(x) > CERO, abs_ / np.abs(x) * 100, abs_ * 100)
        fila = {"Iter": k, **{nombres[i]: x[i] for i in range(n)}}
        for i in range(n):
            fila[f"Err {nombres[i]}" + (" %" if tipo_error != "absoluto" else "")] = (
                rel[i] if tipo_error != "absoluto" else abs_[i]
            )
        filas.append(fila)

        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e14:
            mensaje = f"El metodo diverge (iteracion {k})."
            convergio = False
            break
        error = np.max(rel if tipo_error != "absoluto" else abs_)
        if criterio == "tolerancia" and error <= tol:
            convergio = True
            mensaje = f"Converge en {k} iteraciones (error maximo {error:.3e})."
            break
    else:
        convergio = criterio == "iteraciones"
        mensaje = (f"Se hicieron las {int(max_iter)} iteraciones pedidas." if convergio
                   else f"Se alcanzo el maximo de {int(max_iter)} iteraciones sin converger.")

    return ResultadoSistema(solucion=x, tabla=pd.DataFrame(filas), mensaje=mensaje,
                            convergio=convergio, extra=extra)


# ----------------------------------------------------------- Thomas

def es_tridiagonal(A):
    n = len(A)
    return all(abs(A[i, j]) < CERO for i in range(n) for j in range(n) if abs(i - j) > 1)


def diagonales(A):
    n = len(A)
    a = [0.0] + [A[i, i - 1] for i in range(1, n)]
    bd = [A[i, i] for i in range(n)]
    c = [A[i, i + 1] for i in range(n - 1)] + [0.0]
    return a, bd, c


def thomas(a, bd, c, d, nombres=None):
    """a: subdiagonal (a[0]=0), bd: diagonal, c: superdiagonal (c[-1]=0), d: lado derecho."""
    n = len(d)
    if not (len(a) == len(bd) == len(c) == n):
        raise ErrorDeSistema("Las cuatro listas a, b, c, d deben tener la misma longitud.")
    nombres = nombres or [f"x{i+1}" for i in range(n)]
    a, bd, c, d = (list(map(float, v)) for v in (a, bd, c, d))
    bm, dm = bd.copy(), d.copy()
    filas = [{"i": 1, "a_i": a[0], "b_i": bd[0], "c_i": c[0], "d_i": d[0],
              "w_i": None, "b'_i": bm[0], "d'_i": dm[0]}]
    for i in range(1, n):
        if abs(bm[i - 1]) < CERO:
            raise ErrorDeSistema("Pivote cero en la eliminacion: Thomas no aplica (usa Gauss-Jordan).")
        w = a[i] / bm[i - 1]
        bm[i] = bd[i] - w * c[i - 1]
        dm[i] = d[i] - w * dm[i - 1]
        filas.append({"i": i + 1, "a_i": a[i], "b_i": bd[i], "c_i": c[i], "d_i": d[i],
                      "w_i": w, "b'_i": bm[i], "d'_i": dm[i]})
    if abs(bm[-1]) < CERO:
        raise ErrorDeSistema("Pivote cero: el sistema es singular.")
    x = [0.0] * n
    x[-1] = dm[-1] / bm[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dm[i] - c[i] * x[i + 1]) / bm[i]
    for i in range(n):
        filas[i]["x_i"] = x[i]
    tabla = pd.DataFrame(filas)
    sol = pd.DataFrame({"Incognita": nombres, "Valor": x})
    return ResultadoSistema(solucion=np.array(x), tabla=sol, pasos=[("Barrido", tabla)],
                            mensaje="Eliminacion hacia adelante + sustitucion regresiva.",
                            extra={"b_mod": bm, "d_mod": dm, "a": a, "b": bd, "c": c, "d": d})
