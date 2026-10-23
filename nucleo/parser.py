"""
Convierte lo que escribe el usuario en objetos matematicos.

- funcion de x            ->  f(x) numerica + derivada (SymPy la calcula sola)
- funcion de x e y (EDO)  ->  f(x, y) numerica
- sistema de ecuaciones   ->  matriz A, vector b, nombres de incognitas
- listas de numeros       ->  "0, 2, 4"  o  "0 2 4"  o  "0; 2; 4"
"""

import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X, Y = sp.symbols("x y")

_TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,      # separa 'xy' en x*y (funciones de x, y)
    convert_xor,
)
# en sistemas NO se separan nombres: X1, X2, x_1... son incognitas completas
_TRANSFORMACIONES_SISTEMA = standard_transformations + (
    implicit_multiplication, implicit_application, function_exponentiation, convert_xor,
)

# nombres en espanol -> nombres de SymPy
_ALIAS = {
    r"\bsenh\b": "sinh",
    r"\bsen\b": "sin",
    r"\btg\b": "tan",
    r"\bln\b": "log",
    r"\braiz\b": "sqrt",
}
_FUNCIONES = "asin|acos|atan|sinh|cosh|tanh|senh|sin|sen|cos|tan|tg|exp|log|ln|sqrt|cbrt|raiz|abs|Abs"

_LOCALES = {
    "x": X, "y": Y, "e": sp.E, "E": sp.E, "pi": sp.pi,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "log": sp.log, "sqrt": sp.sqrt, "cbrt": lambda z: sp.real_root(z, 3),
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "Abs": sp.Abs, "abs": sp.Abs,
}

_RESERVADAS = {"e", "E", "pi"}


class ErrorDeEntrada(Exception):
    """Lo que escribio el usuario no se puede interpretar."""


def _limpiar(texto):
    texto = (texto or "").strip()
    texto = texto.replace("√", "sqrt").replace("·", "*").replace("−", "-")
    texto = texto.replace("²", "^2").replace("³", "^3")
    # 2xsen(x) -> 2x*sen(x)   (separa la variable o numero pegado a una funcion)
    texto = re.sub(rf"([0-9xy\)\.])\s*(?=({_FUNCIONES})\s*\()", r"\1*", texto)
    for patron, reemplazo in _ALIAS.items():
        texto = re.sub(patron, reemplazo, texto)
    return texto


def _parsear(texto, locales=None, transformaciones=None):
    texto = _limpiar(texto)
    if not texto:
        raise ErrorDeEntrada("El campo esta vacio.")
    try:
        return parse_expr(texto, local_dict=dict(locales or _LOCALES),
                          transformations=transformaciones or _TRANSFORMACIONES)
    except Exception:
        raise ErrorDeEntrada(f"No pude interpretar «{texto}». Revisa parentesis y signos.")


def _numerica(expr, variables):
    """Convierte una expresion SymPy en funcion de Python que devuelve float."""
    base = sp.lambdify(variables, expr, modules=["numpy"])

    def evaluar(*valores):
        with np.errstate(all="ignore"):
            resultado = base(*valores)
        if np.ndim(resultado) == 0:
            resultado = complex(resultado)
            if abs(resultado.imag) > 1e-12:
                return float("nan")
            return float(resultado.real)
        resultado = np.asarray(resultado, dtype=complex)
        salida = resultado.real.astype(float)
        salida[np.abs(resultado.imag) > 1e-12] = np.nan
        return salida

    return evaluar


def funcion_x(texto):
    """'x^3 - x - 1'  ->  (expr, f, df, expr_derivada)."""
    expr = _parsear(texto)
    sobrantes = expr.free_symbols - {X}
    if sobrantes:
        raise ErrorDeEntrada(
            "La funcion solo puede depender de x (aparece: "
            + ", ".join(sorted(map(str, sobrantes))) + ")."
        )
    f = _numerica(expr, X)
    derivada = sp.diff(expr, X)
    try:
        if derivada.has(sp.Derivative, sp.DiracDelta, sp.Subs):
            raise ValueError
        df = _numerica(derivada, X)
    except Exception:                       # derivada no imprimible -> numerica centrada
        def df(t, h=1e-6):
            return (f(t + h) - f(t - h)) / (2 * h)
    return expr, f, df, derivada


def funcion_xy(texto):
    """'y - x^2 + 1'  ->  (expr, f(x, y))  para dy/dx = f(x, y)."""
    expr = _parsear(texto)
    sobrantes = expr.free_symbols - {X, Y}
    if sobrantes:
        raise ErrorDeEntrada(
            "La EDO solo puede usar x e y (aparece: "
            + ", ".join(sorted(map(str, sobrantes))) + ")."
        )
    return expr, _numerica(expr, (X, Y))


def numero(texto):
    """Acepta '2', '0,5', '1/3', 'pi/4', 'sqrt(2)'."""
    texto = str(texto).strip().replace(",", ".")
    try:
        return float(sp.N(_parsear(texto)))
    except Exception:
        raise ErrorDeEntrada(f"«{texto}» no es un numero valido.")


def lista_numeros(texto):
    """'0, 2, 4'  /  '0 2 4'  /  '0;2;4'  ->  [0.0, 2.0, 4.0].
    Si se usan comas como separador decimal, separa con ';' o espacios."""
    texto = (texto or "").strip()
    if not texto:
        raise ErrorDeEntrada("La lista esta vacia.")
    if ";" in texto:
        partes = texto.split(";")
    elif re.search(r"\d,\d", texto) and " " in texto.strip():
        partes = texto.split()                       # "0,5 1,5 2"  (coma decimal)
    else:
        partes = re.split(r"[,\s]+", texto)
    partes = [p for p in (p.strip() for p in partes) if p]
    return [numero(p) for p in partes]


def puntos(texto):
    """'(0,3)(2,3)(4,0)'  o  una pareja por linea  '0 3'  ->  (xs, ys)."""
    texto = (texto or "").strip()
    pares = re.findall(r"\(\s*([^(),;]+)\s*[,;]\s*([^(),;]+)\s*\)", texto)
    if not pares:
        pares = []
        for linea in texto.splitlines():
            trozos = [t for t in re.split(r"[,;\s]+", linea.strip()) if t]
            if len(trozos) == 2:
                pares.append(tuple(trozos))
            elif trozos:
                raise ErrorDeEntrada(f"La linea «{linea}» debe tener dos valores: x  y.")
    if not pares:
        raise ErrorDeEntrada("No encontre puntos. Escribe (x, y)(x, y)... o una pareja por linea.")
    xs = [numero(a) for a, _ in pares]
    ys = [numero(b) for _, b in pares]
    return xs, ys


# ------------------------------------------------------------- sistemas

def _incognitas_en_orden(texto, simbolos):
    orden = []
    for nombre in re.findall(r"[A-Za-z_]\w*", _limpiar(texto)):
        for s in simbolos:
            if str(s) == nombre and s not in orden:
                orden.append(s)
    for s in sorted(simbolos, key=str):
        if s not in orden:
            orden.append(s)
    return orden


def sistema(texto):
    """
    Una ecuacion por linea:
        2x + y + z = 3
        4x + 3y + 2z = 6
        -2x + 5y + 4z = 7
    Devuelve (A, b, nombres_de_incognitas).
    """
    lineas = [l.strip() for l in (texto or "").splitlines() if l.strip()]
    if not lineas:
        raise ErrorDeEntrada("No escribiste ninguna ecuacion.")

    ecuaciones, simbolos = [], set()
    for numero_linea, linea in enumerate(lineas, start=1):
        if linea.count("=") != 1:
            raise ErrorDeEntrada(
                f"La ecuacion {numero_linea} debe tener exactamente un signo «=»: {linea}"
            )
        izquierda, derecha = linea.split("=")
        locales = {k: v for k, v in _LOCALES.items() if k not in ("x", "y")}
        locales.update({"e": sp.E, "E": sp.E})
        ec = sp.Eq(_parsear(izquierda, locales, _TRANSFORMACIONES_SISTEMA),
                   _parsear(derecha, locales, _TRANSFORMACIONES_SISTEMA))
        ecuaciones.append(ec)
        simbolos |= ec.free_symbols

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
        raise ErrorDeEntrada("El sistema no es lineal (productos entre incognitas, potencias...).")

    A = np.array(A.tolist(), dtype=float)
    b = np.array(b.tolist(), dtype=float).flatten()
    return A, b, [str(s) for s in incognitas]


def sistema_a_texto(A, b, nombres):
    lineas = []
    for fila, termino in zip(A, b):
        partes = []
        for coef, nombre in zip(fila, nombres):
            if abs(coef) < 1e-14:
                continue
            signo = "-" if coef < 0 else ("+" if partes else "")
            mag = abs(coef)
            txt = "" if abs(mag - 1) < 1e-14 else f"{mag:g}"
            partes.append(f"{signo} {txt}{nombre}".strip())
        lineas.append(f"{' '.join(partes) or '0'} = {termino:g}")
    return "\n".join(lineas)


def con_tiempo_limite(funcion, *args, segundos=8, por_defecto=None):
    """Ejecuta funcion(*args); si tarda mas de `segundos`, devuelve por_defecto."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
    ejecutor = ThreadPoolExecutor(max_workers=1)
    futuro = ejecutor.submit(funcion, *args)
    try:
        return futuro.result(timeout=segundos)
    except _Timeout:
        return por_defecto
    except Exception:
        return por_defecto
    finally:
        ejecutor.shutdown(wait=False)
