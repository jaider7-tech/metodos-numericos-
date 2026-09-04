"""
Metodos para resolver sistemas de ecuaciones lineales A x = b.

SEGUNDA IDEA DE LA OPTIMIZACION
-------------------------------
Gauss-Seidel y Gauss-Jacobi se escriben normalmente como dos algoritmos
separados, pero la diferencia entre ellos cabe en una sola linea: Jacobi
usa los valores de la iteracion anterior durante todo el barrido, Seidel
usa los que acaba de calcular.  Aqui hay una unica funcion `_iterativo`
con un interruptor booleano, y dos envoltorios de dos lineas cada uno.
"""

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import pandas as pd

CERO = 1e-12


class ErrorDeMetodo(Exception):
    """El sistema no se puede resolver con el metodo elegido."""


@dataclass
class Resultado:
    solucion: np.ndarray | None = None
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergio: bool = True
    mensaje: str = ""
    pasos: list = field(default_factory=list)      # [(titulo, matriz), ...]
    avisos: list = field(default_factory=list)
    historial: list = field(default_factory=list)  # error por iteracion


def _validar(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ErrorDeMetodo("La matriz de coeficientes debe ser cuadrada.")
    if A.shape[0] != b.size:
        raise ErrorDeMetodo("El numero de ecuaciones no coincide con el de terminos independientes.")
    return A, b


def residuo(A, b, x):
    """Norma del residuo ||Ax - b||: sirve para verificar cualquier solucion."""
    return float(np.linalg.norm(A @ x - b, np.inf))


# ------------------------------------------------------------ Gauss-Jordan

def gauss_jordan(A, b):
    """
    Reduce la matriz aumentada hasta la identidad, con pivoteo parcial.
    Guarda cada matriz intermedia para poder mostrar el procedimiento.
    """
    A, b = _validar(A, b)
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)])
    pasos = [("Matriz aumentada inicial", M.copy())]
    avisos = []

    for i in range(n):
        # pivoteo parcial: se sube la fila con el mayor coeficiente
        fila_pivote = i + int(np.argmax(np.abs(M[i:, i])))
        if abs(M[fila_pivote, i]) < CERO:
            raise ErrorDeMetodo(
                f"La columna {i + 1} no tiene pivote valido: el sistema no tiene "
                "solucion unica (es singular o tiene infinitas soluciones)."
            )
        if fila_pivote != i:
            M[[i, fila_pivote]] = M[[fila_pivote, i]]
            avisos.append(f"Se intercambiaron las filas {i + 1} y {fila_pivote + 1} (pivoteo parcial).")
            pasos.append((f"Intercambio F{i + 1} <-> F{fila_pivote + 1}", M.copy()))

        pivote = M[i, i]
        M[i] = M[i] / pivote
        pasos.append((f"F{i + 1} -> F{i + 1} / {pivote:.6g}", M.copy()))

        for j in range(n):
            if j != i and abs(M[j, i]) > CERO:
                factor = M[j, i]
                M[j] = M[j] - factor * M[i]
                pasos.append((f"F{j + 1} -> F{j + 1} - ({factor:.6g}) F{i + 1}", M.copy()))

    solucion = M[:, -1]
    return Resultado(
        solucion=solucion,
        pasos=pasos,
        avisos=avisos,
        mensaje=f"Solucion exacta obtenida en {len(pasos) - 1} operaciones elementales. "
                f"Residuo ||Ax-b|| = {residuo(A, b, solucion):.2e}",
    )


# ------------------------------------------- Gauss-Seidel y Gauss-Jacobi

def radio_espectral(A, usar_actualizados):
    """
    Radio espectral de la matriz de iteracion.  Es EL criterio de convergencia:
    el metodo converge (desde cualquier punto inicial) si y solo si rho < 1.

        Jacobi        T = -D^-1 (L + U)
        Gauss-Seidel  T = -(D + L)^-1 U
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    try:
        T = -np.linalg.inv(D + L) @ U if usar_actualizados else -np.linalg.inv(D) @ (L + U)
        return float(np.max(np.abs(np.linalg.eigvals(T))))
    except np.linalg.LinAlgError:
        return np.inf


def es_dominante(A):
    n = A.shape[0]
    return all(abs(A[i, i]) >= np.sum(np.abs(A[i])) - abs(A[i, i]) - CERO for i in range(n))


def _orden_voraz(A):
    """Para sistemas grandes: asigna a cada columna la fila de mayor coeficiente."""
    n = A.shape[0]
    M = np.abs(A.copy())
    orden = [None] * n
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(M), M.shape)
        orden[j] = int(i)
        M[i, :] = -1
        M[:, j] = -1
    return orden


def _mejor_orden(A, usar_actualizados):
    """
    Elige la permutacion de ecuaciones que mejor sirve al metodo iterativo.

    En el cuaderno original este reordenamiento estaba hecho a mano y sin
    explicarlo (las ecuaciones estaban emparejadas con formulas de otras
    filas).  Aqui el programa lo hace solo, y ademas justifica la eleccion
    con el radio espectral.
    """
    n = A.shape[0]
    candidatas = list(permutations(range(n))) if n <= 6 else [
        tuple(range(n)), tuple(_orden_voraz(A))
    ]

    mejor, mejor_puntaje = None, None
    for P in candidatas:
        M = A[list(P)]
        if np.min(np.abs(np.diag(M))) < CERO:
            continue                       # cero en la diagonal: inservible
        puntaje = (0 if es_dominante(M) else 1, radio_espectral(M, usar_actualizados))
        if mejor_puntaje is None or puntaje < mejor_puntaje:
            mejor, mejor_puntaje = list(P), puntaje

    if mejor is None:
        raise ErrorDeMetodo(
            "No existe forma de reordenar las ecuaciones para evitar un cero en "
            "la diagonal. Los metodos iterativos no se pueden aplicar a este sistema."
        )
    return mejor, mejor != list(range(n)), mejor_puntaje[0] == 0, mejor_puntaje[1]


def _iterativo(A, b, x0=None, tol=1e-6, max_iter=100, usar_actualizados=False):
    """Motor comun. usar_actualizados=False -> Jacobi ; True -> Gauss-Seidel."""
    A, b = _validar(A, b)
    n = len(b)

    orden, se_reordeno, dominante, rho = _mejor_orden(A, usar_actualizados)
    A, b = A[orden], b[orden]

    avisos = []
    if se_reordeno:
        avisos.append(
            "Se reordenaron las ecuaciones como "
            + " , ".join(f"E{i + 1}" for i in orden)
            + " para que el metodo pueda converger. El sistema es el mismo: "
            "solo cambia el orden en que se escriben las ecuaciones."
        )
    if dominante:
        avisos.append("La matriz es diagonalmente dominante: la convergencia esta garantizada.")
    avisos.append(
        f"Radio espectral de la matriz de iteracion: rho = {rho:.4f}. "
        + ("Como rho < 1, el metodo converge." if rho < 1 else
           "Como rho >= 1, el metodo NO converge con ningun valor inicial.")
    )

    nombres = [f"x{i + 1}" for i in range(n)]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).flatten()
    filas, historial = [], []
    convergio, mensaje = False, ""

    for k in range(1, int(max_iter) + 1):
        anterior = x.copy()
        nuevo = anterior.copy()

        for i in range(n):
            base = nuevo if usar_actualizados else anterior
            suma = A[i] @ base - A[i, i] * base[i]
            nuevo[i] = (b[i] - suma) / A[i, i]

        if not np.all(np.isfinite(nuevo)) or np.max(np.abs(nuevo)) > 1e14:
            mensaje = f"El metodo diverge en la iteracion {k}."
            break

        error = float(np.max(np.abs(nuevo - anterior)))
        porcentual = float(
            np.max(np.abs((nuevo - anterior) / np.where(np.abs(nuevo) < CERO, 1, nuevo))) * 100
        )
        filas.append(
            {"Iter": k, **dict(zip(nombres, nuevo)), "Error abs.": error, "Error %": porcentual}
        )
        historial.append(error)
        x = nuevo

        if error <= tol:
            convergio = True
            mensaje = f"Converge en {k} iteraciones con error {error:.2e}."
            break
    else:
        mensaje = f"Se alcanzo el maximo de {int(max_iter)} iteraciones sin cumplir la tolerancia."

    if convergio:
        mensaje += f" Residuo ||Ax-b|| = {residuo(A, b, x):.2e}"

    return Resultado(
        solucion=x,
        tabla=pd.DataFrame(filas),
        convergio=convergio,
        mensaje=mensaje,
        avisos=avisos,
        historial=historial,
    )


def gauss_jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    """Todas las incognitas se actualizan con los valores de la iteracion previa."""
    return _iterativo(A, b, x0, tol, max_iter, usar_actualizados=False)


def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    """Cada incognita usa de inmediato los valores recien calculados."""
    return _iterativo(A, b, x0, tol, max_iter, usar_actualizados=True)


# ------------------------------------------------------- Algoritmo de Thomas

def diagonales(A):
    """Extrae (subdiagonal, diagonal, superdiagonal) y verifica que sea tridiagonal."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    fuera = [
        (i + 1, j + 1)
        for i in range(n)
        for j in range(n)
        if abs(i - j) > 1 and abs(A[i, j]) > CERO
    ]
    if fuera:
        posiciones = ", ".join(f"({i},{j})" for i, j in fuera[:5])
        raise ErrorDeMetodo(
            "La matriz no es tridiagonal: hay coeficientes distintos de cero en "
            f"{posiciones}. El algoritmo de Thomas solo aplica a matrices tridiagonales."
        )
    inferior = np.array([A[i, i - 1] for i in range(1, n)])
    principal = np.array([A[i, i] for i in range(n)])
    superior = np.array([A[i, i + 1] for i in range(n - 1)])
    return inferior, principal, superior


def thomas(A, b):
    """
    Barrido hacia adelante y sustitucion hacia atras para matrices tridiagonales.
    Cuesta O(n) en lugar de O(n^3): esa es toda su gracia.
    """
    A, b = _validar(A, b)
    inferior, principal, superior = diagonales(A)
    n = len(b)

    c = np.zeros(n)   # superdiagonal modificada
    d = np.zeros(n)   # termino independiente modificado

    if abs(principal[0]) < CERO:
        raise ErrorDeMetodo("El primer elemento de la diagonal es cero: no se puede iniciar el barrido.")
    c[0] = (superior[0] / principal[0]) if n > 1 else 0.0
    d[0] = b[0] / principal[0]

    for i in range(1, n):
        denominador = principal[i] - inferior[i - 1] * c[i - 1]
        if abs(denominador) < CERO:
            raise ErrorDeMetodo(f"Denominador nulo en la fila {i + 1}: el sistema es singular.")
        c[i] = (superior[i] / denominador) if i < n - 1 else 0.0
        d[i] = (b[i] - inferior[i - 1] * d[i - 1]) / denominador

    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    tabla = pd.DataFrame({
        "Fila": range(1, n + 1),
        "a (inferior)": np.concatenate([[np.nan], inferior]),
        "b (diagonal)": principal,
        "c (superior)": np.concatenate([superior, [np.nan]]),
        "d (indep.)": b,
        "c' modificada": c,
        "d' modificada": d,
        "Solucion": x,
    })

    return Resultado(
        solucion=x,
        tabla=tabla,
        mensaje=f"Resuelto en un solo barrido (O(n) operaciones). "
                f"Residuo ||Ax-b|| = {residuo(A, b, x):.2e}",
    )
