"""
app.py  -  Interfaz (tkinter) que une todos los modulos.
Ejecutar:  python app.py
Instalar:  pip install numpy matplotlib
"""
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from arbol_expresion import ArbolExpresion               # clase principal (sin cambios)
from generador_aleatorio import generar_graficable, altura  # ademas carga la biblioteca de operadores
from visualizacion import (ROSA, BLANCO, NEGRO, dibujar_arbol, dibujar_funcion)


class App:
    def __init__(self, ventana):
        ventana.title("Arbol de expresion")
        ventana.configure(bg=BLANCO)

        # --- Barra superior con los inputs ---
        barra = tk.Frame(ventana, bg=ROSA, padx=10, pady=8)
        barra.pack(fill="x")
        estilo = dict(bg=ROSA, fg=BLANCO, font=("Arial", 11, "bold"))

        # Fila 1: expresion
        tk.Label(barra, text="Expresion:", **estilo).grid(row=0, column=0, sticky="w")
        self.entrada_expr = tk.Entry(barra, width=60, font=("Arial", 12))
        self.entrada_expr.insert(0, "sqrt(2 * (x ^2 ) + 5 * x - 3)")
        self.entrada_expr.grid(row=0, column=1, columnspan=6, padx=8, sticky="w")

        # Fila 2: intervalo, altura y botones
        tk.Label(barra, text="Intervalo (a, b):", **estilo).grid(row=1, column=0, sticky="w", pady=6)
        self.entrada_int = tk.Entry(barra, width=10, font=("Arial", 12))
        self.entrada_int.insert(0, "-10, 10")
        self.entrada_int.grid(row=1, column=1, padx=8, sticky="w")

        tk.Label(barra, text="Altura max:", **estilo).grid(row=1, column=2, sticky="e")
        self.entrada_alt = tk.Entry(barra, width=5, font=("Arial", 12))
        self.entrada_alt.insert(0, "4")
        self.entrada_alt.grid(row=1, column=3, padx=8)

        boton = dict(bg=NEGRO, fg=BLANCO, font=("Arial", 11, "bold"), relief="flat", padx=14)
        tk.Button(barra, text="Graficar", command=self.graficar, **boton).grid(row=1, column=4, padx=6)
        tk.Button(barra, text="Aleatoria", command=self.aleatoria, **boton).grid(row=1, column=5, padx=6)

        # --- Texto: preorden y altura ---
        self.etiqueta = tk.Label(ventana, text="", bg=BLANCO, fg=NEGRO,
                                 font=("Consolas", 11), wraplength=1100, justify="left")
        self.etiqueta.pack(fill="x", padx=10, pady=6)

        # --- Figura con 2 graficas: arbol y funcion ---
        self.fig = Figure(figsize=(12, 5), facecolor=BLANCO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=ventana)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.graficar()

    # ---- lectura de inputs ----
    def _intervalo(self):
        a, b = [float(v) for v in self.entrada_int.get().split(",")]
        return a, b

    def _altura_max(self):
        return int(self.entrada_alt.get())

    # ---- botones ----
    def graficar(self):
        """Usa la expresion escrita en el input."""
        try:
            self.mostrar(ArbolExpresion(self.entrada_expr.get()))
        except Exception as error:
            messagebox.showerror("Error", f"Revisa tus datos:\n{error}")

    def aleatoria(self):
        """Genera una funcion aleatoria respetando la altura maxima."""
        try:
            a, b = self._intervalo()
            arbol = generar_graficable(self._altura_max(), a, b)
            self.entrada_expr.delete(0, "end")
            self.entrada_expr.insert(0, arbol.expresion)
            self.mostrar(arbol)
        except Exception as error:
            messagebox.showerror("Error", f"Revisa tus datos:\n{error}")

    # ---- dibujo ----
    def mostrar(self, arbol):
        a, b = self._intervalo()
        h, h_max = altura(arbol.raiz), self._altura_max()
        self.fig.clear()
        dibujar_arbol(self.fig.add_subplot(1, 2, 1), arbol)
        dibujar_funcion(self.fig.add_subplot(1, 2, 2), arbol, a, b)
        self.fig.tight_layout()
        self.canvas.draw()
        aviso = "   (excede la altura maxima)" if h > h_max else ""
        self.etiqueta.config(text=f"Altura: {h} / {h_max}{aviso}\n"
                                  f"Preorden: " + "  ".join(arbol.preorden()))


if __name__ == "__main__":
    ventana = tk.Tk()
    App(ventana)
    ventana.mainloop()
