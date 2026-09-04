"""
Traduce lo que el usuario escribe en texto a objetos matematicos usables.

Aqui esta la razon por la que la aplicacion acepta CUALQUIER ejercicio:
en vez de tener la funcion escrita a mano dentro del codigo, se interpreta
lo que el usuario teclea usando SymPy.  Ademas SymPy sabe derivar, asi que
Newton-Raphson no necesita que nadie escriba f'(x).
"""

import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X = sp.Symbol("x")

# convert_xor  ->  permite escribir  x^2  en vez de  x**2
# implicit_... ->  permite escribir  2x   en vez de  2*x   y   sin x
_TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_IDENTIFICADOR = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

# Nombres que son funciones, no incognitas.
_RESERVADAS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "exp", "log", "ln", "sqrt", "abs", "pi", "E", "e",
}


class ErrorDeEntrada(Exception):
    """Se lanza cuando el usuario escribio algo que no se puede interpretar."""


def _parsear(texto):
    texto = (texto or "").strip()
    if not texto:
        raise ErrorDeEntrada("El campo esta vacio.")
    try:
        return parse_expr(texto, transformations=_TRANSFORMACIONES)
    except Exception as error:
        raise ErrorDeEntrada(f"No pude interpretar «{texto}». Detalle: {error}")


# ---------------------------------------------------------------- funciones

def funcion(texto, variable=X):
    """Devuelve (f_numerica, expresion_simbolica) a partir de un texto."""
    expresion = _parsear(texto)
    sobrantes = expresion.free_symbols - {variable}
    if sobrantes:
        nombres = ", ".join(sorted(str(s) for s in sobrantes))
        raise ErrorDeEntrada(
            f"La expresion usa la(s) variable(s) «{nombres}». "
            f"Solo se admite «{variable}»."
        )
    return sp.lambdify(variable, expresion, "numpy"), expresion


def derivada(expresion, variable=X):
    """Deriva simbolicamente y devuelve (f'_numerica, expresion_derivada)."""
    d = sp.diff(expresion, variable)
    return sp.lambdify(variable, d, "numpy"), d


def evaluar(f, xs):
    """Evalua f sobre un arreglo tolerando dominios invalidos (log, sqrt...)."""
    salida = np.full(len(xs), np.nan, dtype=float)
    for i, xi in enumerate(xs):
        try:
            valor = float(f(xi))
            if np.isfinite(valor):
                salida[i] = valor
        except Exception:
            pass
    return salida


def bonito(expresion):
    """Version LaTeX para mostrar la expresion en pantalla."""
    return sp.latex(expresion)


# ------------------------------------------------------- sistemas lineales

def _incognitas_en_orden(texto, simbolos):
    """Ordena las incognitas segun su primera aparicion en el texto."""
    vistos, orden = set(), []
    nombres = {str(s): s for s in simbolos}
    for nombre in _IDENTIFICADOR.findall(texto):
        if nombre in nombres and nombre not in vistos:
            vistos.add(nombre)
            orden.append(nombres[nombre])
    # por si alguna incognita solo aparece en una forma rara
    for s in simbolos:
        if s not in orden:
            orden.append(s)
    return orden


def sistema_desde_texto(texto):
    """
    Convierte un sistema escrito en lenguaje natural a matrices (A, b).

    Ejemplo de entrada:
        2x + y + z = 3
        4x + 3y + 2z = 6
        -2x + 5y + 4z = 7

    Devuelve (A, b, nombres_de_incognitas).
    """
    lineas = [l.strip() for l in (texto or "").splitlines() if l.strip()]
    if not lineas:
        raise ErrorDeEntrada("No escribiste ninguna ecuacion.")

    ecuaciones, simbolos = [], set()
    for numero, linea in enumerate(lineas, start=1):
        if linea.count("=") != 1:
            raise ErrorDeEntrada(
                f"La ecuacion {numero} debe tener exactamente un signo «=»: {linea}"
            )
        izquierda, derecha = linea.split("=")
        ecuacion = sp.Eq(_parsear(izquierda), _parsear(derecha))
        ecuaciones.append(ecuacion)
        simbolos |= ecuacion.free_symbols

    simbolos = {s for s in simbolos if str(s) not in _RESERVADAS}
    incognitas = _incognitas_en_orden(texto, simbolos)

    if len(ecuaciones) != len(incognitas):
        raise ErrorDeEntrada(
            f"Hay {len(ecuaciones)} ecuacion(es) y {len(incognitas)} incognita(s). "
            "El sistema debe ser cuadrado."
        )

    try:
        A, b = sp.linear_eq_to_matrix(ecuaciones, incognitas)
    except Exception:
        raise ErrorDeEntrada(
            "El sistema no es lineal (hay productos entre incognitas, potencias, etc.)."
        )

    A = np.array(A.tolist(), dtype=float)
    b = np.array(b.tolist(), dtype=float).flatten()
    return A, b, [str(s) for s in incognitas]


def sistema_a_texto(A, b, nombres):
    """Reconstruye las ecuaciones en texto (util para mostrar el sistema)."""
    lineas = []
    for fila, termino in zip(A, b):
        partes = []
        for coeficiente, nombre in zip(fila, nombres):
            if abs(coeficiente) < 1e-14:
                continue
            signo = "-" if coeficiente < 0 else ("+" if partes else "")
            magnitud = abs(coeficiente)
            texto = "" if abs(magnitud - 1) < 1e-14 else f"{magnitud:g}"
            partes.append(f"{signo} {texto}{nombre}".strip())
        lineas.append(f"{' '.join(partes) or '0'} = {termino:g}")
    return "\n".join(lineas)
