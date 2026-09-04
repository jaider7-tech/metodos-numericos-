# ∑ Métodos Numéricos — aplicación web

Aplicación web que resuelve **cualquier** ecuación o sistema lineal que el usuario
escriba, mostrando la tabla de iteraciones y las gráficas de cada método.

Construida en Python con Streamlit, SymPy, NumPy, pandas y Matplotlib.

---

## Métodos incluidos

**Raíces de ecuaciones**

| Método | Tipo | Qué necesita |
|---|---|---|
| Bisección | cerrado | intervalo `[a, b]` con cambio de signo |
| Punto fijo | abierto | `g(x)` y un valor inicial |
| Regula Falsi | cerrado | intervalo `[a, b]` con cambio de signo |
| Newton-Raphson | abierto | un valor inicial (la derivada se calcula sola) |
| Secante | abierto | dos valores iniciales |

Incluye además un modo **Comparar los 5 métodos**, que corre todos sobre la misma
función y muestra cuál converge en menos iteraciones.

**Sistemas de ecuaciones lineales**

| Método | Tipo | Observación |
|---|---|---|
| Gauss-Jordan | directo | muestra todas las operaciones elementales |
| Gauss-Seidel | iterativo | reordena las ecuaciones automáticamente |
| Gauss-Jacobi | iterativo | reordena las ecuaciones automáticamente |
| Algoritmo de Thomas | directo | solo matrices tridiagonales, O(n) |

---

## Cómo ejecutarla en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre sola en el navegador, en `http://localhost:8501`.

Para verificar que todos los métodos dan resultados correctos:

```bash
python pruebas.py
```

Compara cada método contra el valor exacto o contra `numpy.linalg.solve`.

---

## Cómo publicarla en internet (gratis)

1. Crea una cuenta en [github.com](https://github.com) si no la tienes.
2. Crea un repositorio **público** nuevo, por ejemplo `metodos-numericos-app`.
3. Sube estos archivos al repositorio (botón *Add file → Upload files*, arrastra
   la carpeta completa incluyendo `nucleo/`).
4. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
5. Clic en *New app*, escoge tu repositorio, rama `main` y archivo principal `app.py`.
6. Clic en *Deploy*. En unos dos minutos queda un enlace del tipo
   `https://tu-usuario-metodos-numericos.streamlit.app`

Ese enlace abre en cualquier navegador, celular incluido, sin instalar nada.

---

## Cómo escribir las expresiones

| Quieres escribir | Escribe |
|---|---|
| x³ − x − 1 | `x^3 - x - 1` o `x**3 - x - 1` |
| 2x² + 5x | `2x^2 + 5x` |
| sen(x) − x/2 | `sin(x) - x/2` |
| e^x − 3 | `exp(x) - 3` |
| √(x + 2) | `sqrt(x + 2)` |
| ln(x) | `log(x)` |

Los sistemas se escriben una ecuación por línea, y las incógnitas pueden llamarse
como quieras:

```
2x + y + z = 3
4x + 3y + 2z = 6
-2x + 5y + 4z = 7
```

---

## Estructura del proyecto

```
app.py                 interfaz web (solo pide datos y dibuja)
nucleo/
  parser.py            convierte texto en funciones y matrices
  raices.py            los 5 métodos de raíces
  sistemas.py          los 4 métodos de sistemas lineales
pruebas.py             verificación automática contra NumPy
requirements.txt       dependencias
```

El cálculo está separado de la interfaz: `nucleo/` funciona por sí solo, sin
Streamlit. Por eso `pruebas.py` puede verificarlo sin abrir la aplicación.

---

## Qué se mejoró respecto al cuaderno original

El archivo de partida tenía **3.416 líneas** organizadas como ejercicios sueltos:
cada método se reescribía completo por cada ejemplo, con la función y los datos
incrustados en el código. Para cambiar de ejercicio había que editar el código
fuente.

| | Cuaderno original | Esta aplicación |
|---|---|---|
| Líneas de código | 3.416 | ~950 (nueve métodos, interfaz y pruebas incluidas) |
| Veces que se define `f(x)` | 23 | 0 (la escribe el usuario) |
| `import numpy` repetidos | 41 | 1 por módulo |
| `print()` sueltos | 231 | 0 (tablas y gráficas) |
| Ejercicios que resuelve | los que están escritos | cualquiera |

**Las tres ideas de fondo:**

1. **Un solo motor de iteración para los cinco métodos de raíces.** Bisección,
   punto fijo, regula falsi, Newton y secante comparten el mismo bucle
   (`raices._ejecutar`); lo único que cambia es la regla que calcula la siguiente
   aproximación, y cada regla ocupa entre cuatro y ocho líneas.

2. **Gauss-Seidel y Gauss-Jacobi son la misma función.** La diferencia cabe en un
   parámetro booleano: Jacobi usa los valores de la iteración anterior durante
   todo el barrido, Seidel usa los que acaba de calcular
   (`sistemas._iterativo(..., usar_actualizados)`).

3. **El programa reordena las ecuaciones solo.** En el cuaderno original, el
   sistema de Gauss-Seidel aparecía con las ecuaciones emparejadas con fórmulas de
   otras filas: el reordenamiento estaba hecho a mano y sin explicarlo. Sin ese
   reordenamiento el método **diverge** (radio espectral ρ = 2.94). Aquí el
   programa prueba las permutaciones de ecuaciones, escoge la que minimiza el
   radio espectral de la matriz de iteración (ρ = 0.17 para ese sistema), avisa
   del cambio y justifica por qué el método va a converger.
