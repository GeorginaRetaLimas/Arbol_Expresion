"""
app.py  -  Interfaz gráfica (Tkinter) que integra todos los módulos.
Ejecutar:  python app.py
Instalar:  pip install numpy matplotlib

Nuevas funcionalidades añadidas:
1. 🔍 Vista ampliada / Pantalla completa de la gráfica:
   - Opciones: Solo función del árbol, Solo función del CSV, Ambas funciones juntas.
   - Visualización de márgenes de error calculados (RMSE, MSE, NUSE, R^2, ÿ).
   - Barra de herramientas interactiva de matplotlib (zoom, paneo, guardar).
2. 🧬 Apartado de Algoritmo Genético (Regresión Simbólica):
   - Estructura de Individuo: {gen: Arbol, feno: valores, adap: margen de error}.
   - Población de 100 ecuaciones aleatorias.
   - Selección por Ruleta (y opción de Torneo).
   - Cruza de subárboles entre individuos.
   - Mutación al 1% con los 4 métodos (Hojas, Nodos internos, Switching, Nueva rama)
     indicando explícitamente en el registro cuál ocurrió.
   - Sincronización visual en vivo de la gráfica con el CSV conforme evoluciona.
   - Botón para ver el Árbol de Expresión Final resultante.
"""
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from arbol_expresion import ArbolExpresion
from generador_aleatorio import generar_graficable, altura
from visualizacion import (
    ROSA, ROSA_CLARO, ROSA_MEDIO, ROSA_BORDE, BLANCO, NEGRO, TEXTO_MUTED,
    AZUL_OBJETIVO, GRIS_REJILLA, dibujar_arbol, dibujar_funcion
)
from metricas import (
    EXPRESION_OBJETIVO, ARCHIVO_CSV_DEFAULT,
    cargar_csv, evaluar_arbol_contra_datos, generar_csv, calcular_metricas
)
from algoritmo_genetico import AlgoritmoGenetico, Individuo

# Paleta cromática estética rosita suave
COLOR_FONDO_APP = "#faf3f7"         # Fondo suave aperlado
COLOR_TARJETA = "#ffffff"           # Tarjetas blancas limpias
COLOR_BORDE_TARJETA = "#edd4e2"     # Borde pastel suave
COLOR_HEADER = "#f8e1ec"            # Banner superior rosa pastel
COLOR_TITULO = "#381a2c"            # Ciruela profundo para títulos principales
COLOR_SUBTITULO = "#754f66"         # Ciruela medio para subtítulos y etiquetas
COLOR_BTN_PRIMARIO = "#d94a7a"      # Rosa frambuesa principal
COLOR_BTN_PRIMARIO_HOVER = "#c53b6b"# Hover rosa intenso
COLOR_BTN_SECUNDARIO = "#faebf2"    # Botón rosa pastel claro
COLOR_BTN_SECUNDARIO_TX = "#a83260" # Texto de botón secundario
COLOR_BTN_EVALUAR = "#993d6b"       # Ciruela/frambuesa oscuro para botón de evaluación
COLOR_BTN_EVALUAR_HOVER = "#852e59" # Hover botón evaluar
COLOR_METRICA_BG = "#fcf2f6"        # Fondo para cajitas de métricas
COLOR_METRICA_BORDE = "#ebd0df"     # Borde para cajitas de métricas


# =====================================================================
# VENTANA MODAL: VISTA AMPLIADA / PANTALLA COMPLETA DE LA GRÁFICA
# =====================================================================
class VentanaVistaAmpliada(tk.Toplevel):
    """
    Permite una vista completa y espaciosa de la gráfica con:
    - Solo función f(x)
    - Solo función del CSV
    - Ambas funciones juntas
    - Valores de margen de error calculados
    - Barra de herramientas (zoom, paneo)
    """
    def __init__(self, parent, arbol, x_csv, y_csv, a=-10.0, b=10.0):
        super().__init__(parent)
        self.title("🔍 Vista Ampliada de Gráfica • Pantalla Completa")
        self.geometry("1280x820")
        self.minsize(950, 600)
        self.configure(bg=COLOR_FONDO_APP)

        self.arbol = arbol
        self.x_csv = x_csv
        self.y_csv = y_csv
        self.a = a
        self.b = b

        self.modo_vista = tk.StringVar(value="ambas")  # 'arbol', 'csv', 'ambas'
        self.es_pantalla_completa = False

        self._crear_interfaz()
        self.actualizar_grafica()

    def _crear_interfaz(self):
        # 1. Barra de controles superior
        barra_sup = tk.Frame(self, bg=COLOR_TARJETA, padx=16, pady=10,
                             highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        barra_sup.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(barra_sup, text="Vista:", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 8))

        opciones = [
            ("🌸 Ambas funciones juntas", "ambas"),
            ("📈 Solo función f(x) del Árbol", "arbol"),
            ("📊 Solo función del CSV (Objetivo)", "csv"),
        ]
        for texto, valor in opciones:
            rb = tk.Radiobutton(
                barra_sup, text=texto, variable=self.modo_vista, value=valor,
                bg=COLOR_TARJETA, fg=COLOR_TITULO, activebackground=COLOR_TARJETA,
                selectcolor=COLOR_TARJETA, font=("Segoe UI", 9, "bold"),
                command=self.actualizar_grafica
            )
            rb.pack(side="left", padx=10)

        # Controles de intervalo
        tk.Label(barra_sup, text="Rango [a, b]:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(20, 6))
        self.ent_a = tk.Entry(barra_sup, width=6, font=("Segoe UI", 9), justify="center",
                              bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_a.insert(0, str(self.a))
        self.ent_a.pack(side="left", padx=2)

        tk.Label(barra_sup, text="a", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO).pack(side="left", padx=2)

        self.ent_b = tk.Entry(barra_sup, width=6, font=("Segoe UI", 9), justify="center",
                              bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_b.insert(0, str(self.b))
        self.ent_b.pack(side="left", padx=2)

        btn_act = tk.Button(barra_sup, text="Actualizar Rango", bg=COLOR_BTN_SECUNDARIO,
                            fg=COLOR_BTN_SECUNDARIO_TX, font=("Segoe UI", 8, "bold"),
                            relief="flat", cursor="hand2", padx=8, pady=2,
                            command=self._leer_intervalo_y_actualizar)
        btn_act.pack(side="left", padx=8)

        # Botón de alternar pantalla completa
        btn_fs = tk.Button(barra_sup, text="⛶ Pantalla Completa", bg=COLOR_BTN_PRIMARIO,
                           fg=BLANCO, font=("Segoe UI", 8, "bold"),
                           relief="flat", cursor="hand2", padx=10, pady=2,
                           command=self._alternar_pantalla_completa)
        btn_fs.pack(side="right", padx=6)

        # 2. Banner de márgenes de error calculados
        self.banner_metricas = tk.Frame(self, bg=COLOR_METRICA_BG, padx=16, pady=8,
                                        highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
        self.banner_metricas.pack(fill="x", padx=12, pady=4)

        self.lbl_met_texto = tk.Label(self.banner_metricas, text="Calculando métricas...",
                                      bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                                      font=("Consolas", 10, "bold"))
        self.lbl_met_texto.pack(side="left")

        # 3. Canvas matplotlib a pantalla completa
        frame_plot = tk.Frame(self, bg=COLOR_TARJETA,
                              highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_plot.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        self.fig = Figure(figsize=(12, 7), facecolor=BLANCO)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Barra de navegación matplotlib (zoom interactivo, paneo, guardar)
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame_plot)
        self.toolbar.config(background=COLOR_TARJETA)
        self.toolbar.update()

    def _alternar_pantalla_completa(self):
        self.es_pantalla_completa = not self.es_pantalla_completa
        self.attributes("-fullscreen", self.es_pantalla_completa)

    def _leer_intervalo_y_actualizar(self):
        try:
            self.a = float(self.ent_a.get().strip())
            self.b = float(self.ent_b.get().strip())
            if self.a >= self.b:
                raise ValueError("a debe ser menor que b")
            self.actualizar_grafica()
        except Exception as e:
            messagebox.showerror("Error de Rango", str(e), parent=self)

    def actualizar_grafica(self):
        self.ax.clear()
        self.ax.set_facecolor(BLANCO)
        modo = self.modo_vista.get()

        # 1. Graficar CSV si corresponde
        if modo in ("csv", "ambas") and self.x_csv is not None and self.y_csv is not None:
            mascara = (self.x_csv >= self.a) & (self.x_csv <= self.b)
            if np.any(mascara):
                x_sub = self.x_csv[mascara]
                y_sub = self.y_csv[mascara]
                if len(x_sub) > 2500:
                    paso = len(x_sub) // 2000
                    x_sub = x_sub[::paso]
                    y_sub = y_sub[::paso]
                self.ax.plot(x_sub, y_sub, color=AZUL_OBJETIVO, lw=2.2, ls="--",
                             label="Función del CSV (Objetivo)", alpha=0.9, zorder=2)

        # 2. Graficar Árbol si corresponde
        if modo in ("arbol", "ambas") and self.arbol is not None:
            x_pts = np.linspace(self.a, self.b, 1200)
            with np.errstate(all="ignore"):
                y_pts = np.asarray(self.arbol.evaluar(x_pts), dtype=float) * np.ones_like(x_pts)
            y_pts[np.abs(y_pts) > 1e6] = np.nan
            self.ax.plot(x_pts, y_pts, color=ROSA, lw=2.6, label="Función del Árbol f(x)", zorder=3)

        # Ejes y rejilla
        self.ax.axhline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
        self.ax.axvline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
        self.ax.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.2, zorder=0)

        for spine in self.ax.spines.values():
            spine.set_color("#e3c9d6")
            spine.set_linewidth(1.2)

        self.ax.tick_params(colors=TEXTO_MUTED, labelsize=10)
        self.ax.set_title(f"Vista Ampliada de Funciones en [{self.a}, {self.b}]",
                          color=NEGRO, fontsize=13, fontweight="bold", pad=12)
        self.ax.set_xlabel("x", color=NEGRO, fontsize=11, fontweight="bold")
        self.ax.set_ylabel("f(x)", color=NEGRO, fontsize=11, fontweight="bold")

        if modo == "ambas":
            self.ax.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=10, loc="best")
        elif modo == "csv":
            self.ax.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=10, loc="best")
        else:
            self.ax.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=10, loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

        # 3. Calcular y mostrar valores de margen de error
        if self.arbol is not None and self.x_csv is not None and self.y_csv is not None:
            try:
                _, met = evaluar_arbol_contra_datos(self.arbol, self.x_csv, self.y_csv)
                if met["valido"]:
                    txt = (
                        f"Margen de Error:  RMSE = {met['rmse']:.6f}  |  MSE = {met['mse']:.6f}  |  "
                        f"NUSE / NMSE = {met['nmse']:.6f}  |  R² = {met['r2']:.4f}  |  "
                        f"ÿ = {met['y_barra']:.4f}  |  Puntos: {met['n_validos']:,} / {met['n_total']:,}"
                    )
                else:
                    txt = "Margen de Error: Indefinido (valores infinitos o no numéricos)"
            except Exception as e:
                txt = f"Error al calcular métricas: {e}"
        else:
            txt = "Sin dataset CSV cargado para calcular margen de error"

        self.lbl_met_texto.config(text=txt)


# =====================================================================
# VENTANA MODAL: VISUALIZACIÓN DEL ÁRBOL DE EXPRESIÓN FINAL
# =====================================================================
class VentanaArbolFinal(tk.Toplevel):
    """Muestra el árbol binario resultante del algoritmo genético."""
    def __init__(self, parent, arbol, titulo_extra=""):
        super().__init__(parent)
        self.title(f"🌳 Estructura del Árbol de Expresión Final {titulo_extra}")
        self.geometry("1050x700")
        self.minsize(800, 550)
        self.configure(bg=COLOR_FONDO_APP)

        header = tk.Frame(self, bg=COLOR_HEADER, padx=16, pady=10,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        header.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(header, text="🌳 Árbol de Expresión Binario Final",
                 bg=COLOR_HEADER, fg=COLOR_TITULO, font=("Segoe UI", 14, "bold")).pack(anchor="w")

        expr_txt = arbol.expresion if hasattr(arbol, "expresion") else str(arbol)
        h = altura(arbol.raiz)
        tk.Label(header, text=f"Expresión: {expr_txt}  •  Altura: {h}",
                 bg=COLOR_HEADER, fg=COLOR_SUBTITULO, font=("Consolas", 10)).pack(anchor="w")

        frame_canvas = tk.Frame(self, bg=COLOR_TARJETA,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_canvas.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        fig = Figure(figsize=(10, 6), facecolor=BLANCO)
        ax = fig.add_subplot(1, 1, 1)
        dibujar_arbol(ax, arbol)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_canvas)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        canvas.draw()


# =====================================================================
# CLASE PRINCIPAL: APLICACIÓN CON PESTAÑAS
# =====================================================================
class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("🌸 Árbol de Expresión Binario & Algoritmo Genético")
        self.ventana.geometry("1280x880")
        self.ventana.minsize(1080, 740)
        self.ventana.configure(bg=COLOR_FONDO_APP)

        # Cargar datos del CSV
        self.ruta_csv = ARCHIVO_CSV_DEFAULT
        self.x_csv = None
        self.y_csv = None
        self._cargar_datos_csv()

        # Variables de estado del analizador
        self.evaluado_con_csv = False
        self.arbol_actual = None

        # Variables del algoritmo genético
        self.ga = None
        self.ga_corriendo = False
        self.ga_timer_id = None
        self.max_generaciones = 100

        self._crear_interfaz_principal()
        self.graficar()

    def _cargar_datos_csv(self):
        try:
            if not os.path.exists(self.ruta_csv):
                generar_csv(self.ruta_csv)
            self.x_csv, self.y_csv = cargar_csv(self.ruta_csv)
        except Exception as e:
            self.x_csv, self.y_csv = None, None
            print(f"Aviso CSV: {e}")

    def _crear_interfaz_principal(self):
        # 1. ENCABEZADO SUPERIOR
        header = tk.Frame(self.ventana, bg=COLOR_HEADER, pady=10, padx=16,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        header.pack(fill="x", padx=12, pady=(10, 6))

        titulo_box = tk.Frame(header, bg=COLOR_HEADER)
        titulo_box.pack(side="left")

        tk.Label(titulo_box, text="🌸 Árbol de Expresión Binario & Regresión Genética",
                 bg=COLOR_HEADER, fg=COLOR_TITULO, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(titulo_box,
                 text="Modelado simbólico jerárquico • Algoritmo genético con mutación (1%) • Métricas del pizarrón",
                 bg=COLOR_HEADER, fg=COLOR_SUBTITULO, font=("Segoe UI", 9)).pack(anchor="w")

        # 2. SISTEMA DE PESTAÑAS (NOTEBOOK)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLOR_FONDO_APP, borderwidth=0)
        style.configure("TNotebook.Tab", background=COLOR_HEADER, foreground=COLOR_TITULO,
                        font=("Segoe UI", 10, "bold"), padding=[20, 7])
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_TARJETA)],
                  foreground=[("selected", COLOR_BTN_PRIMARIO)])

        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        # Pestaña 1: Analizador de Árbol
        self.tab_analizador = tk.Frame(self.notebook, bg=COLOR_FONDO_APP)
        self.notebook.add(self.tab_analizador, text="  🌸 Analizador & Árbol Binario  ")

        # Pestaña 2: Algoritmo Genético
        self.tab_genetico = tk.Frame(self.notebook, bg=COLOR_FONDO_APP)
        self.notebook.add(self.tab_genetico, text="  🧬 Algoritmo Genético (Regresión)  ")

        self._construir_pestana_analizador()
        self._construir_pestana_genetico()

    # =========================================================================
    # CONSTRUCCIÓN PESTAÑA 1: ANALIZADOR DE ÁRBOL
    # =========================================================================
    def _construir_pestana_analizador(self):
        # Panel de control
        panel = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA, padx=14, pady=8,
                         highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        panel.pack(fill="x", pady=(6, 4))

        # Fila 1: Expresion
        f1 = tk.Frame(panel, bg=COLOR_TARJETA)
        f1.pack(fill="x", pady=(2, 6))

        tk.Label(f1, text="Expresión f(x):", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 10, "bold"), width=13, anchor="w").pack(side="left")

        self.entrada_expr = tk.Entry(f1, font=("Consolas", 11), bg="#fffafc", fg=NEGRO,
                                     highlightbackground=COLOR_BORDE_TARJETA,
                                     highlightcolor=COLOR_BTN_PRIMARIO,
                                     highlightthickness=1, relief="flat")
        self.entrada_expr.insert(0, EXPRESION_OBJETIVO)
        self.entrada_expr.pack(side="left", fill="x", expand=True, padx=(4, 10), ipady=4)

        btn_obj = tk.Button(f1, text="🎯 Cargar Objetivo CSV", bg=COLOR_TARJETA,
                            fg=COLOR_BTN_PRIMARIO, activebackground="#faebf2",
                            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                            highlightbackground=COLOR_BTN_PRIMARIO, highlightthickness=1,
                            padx=10, pady=2, command=self.cargar_expresion_objetivo)
        btn_obj.pack(side="right")

        # Fila 2: Parametros y botones
        f2 = tk.Frame(panel, bg=COLOR_TARJETA)
        f2.pack(fill="x", pady=2)

        tk.Label(f2, text="Intervalo [a, b]:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_int = tk.Entry(f2, width=12, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_int.insert(0, "-10, 10")
        self.entrada_int.pack(side="left", padx=(6, 16), ipady=2)

        tk.Label(f2, text="Altura máx:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_alt = tk.Entry(f2, width=5, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_alt.insert(0, "4")
        self.entrada_alt.pack(side="left", padx=(6, 16), ipady=2)

        # Botones
        btn_prim = dict(bg=COLOR_BTN_PRIMARIO, fg=BLANCO, activebackground=COLOR_BTN_PRIMARIO_HOVER,
                        activeforeground=BLANCO, font=("Segoe UI", 9, "bold"), relief="flat",
                        cursor="hand2", padx=12, pady=3)
        btn_sec = dict(bg=COLOR_BTN_SECUNDARIO, fg=COLOR_BTN_SECUNDARIO_TX,
                       activebackground="#f3d3e2", activeforeground=COLOR_BTN_SECUNDARIO_TX,
                       font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                       highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1,
                       padx=10, pady=3)
        btn_eval = dict(bg=COLOR_BTN_EVALUAR, fg=BLANCO, activebackground=COLOR_BTN_EVALUAR_HOVER,
                        activeforeground=BLANCO, font=("Segoe UI", 9, "bold"), relief="flat",
                        cursor="hand2", padx=12, pady=3)

        tk.Button(f2, text="📈 Graficar", command=self.graficar, **btn_prim).pack(side="left", padx=3)
        tk.Button(f2, text="🎲 Aleatoria", command=self.aleatoria, **btn_sec).pack(side="left", padx=3)

        sep = tk.Frame(f2, bg=COLOR_BORDE_TARJETA, width=1, height=22)
        sep.pack(side="left", padx=6, fill="y")

        # Botón de evaluación bajo demanda
        tk.Button(f2, text="📊 Evaluar con CSV", command=self.evaluar_con_csv, **btn_eval).pack(side="left", padx=3)

        # Botón de vista ampliada
        tk.Button(f2, text="🔍 Vista Ampliada Gráfica", command=self.abrir_vista_ampliada,
                  bg="#6a4c93", fg=BLANCO, activebackground="#583c7f", activeforeground=BLANCO,
                  font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=12, pady=3).pack(side="left", padx=3)

        tk.Button(f2, text="📁 Seleccionar CSV...", command=self.seleccionar_csv, **btn_sec).pack(side="left", padx=3)

        # Card de Info del Árbol
        self.card_info = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA, padx=14, pady=6,
                                  highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.card_info.pack(fill="x", pady=4)

        info_top = tk.Frame(self.card_info, bg=COLOR_TARJETA)
        info_top.pack(fill="x")

        self.lbl_altura = tk.Label(info_top, text="Altura: -", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_altura.pack(side="left")

        self.lbl_preorden = tk.Label(info_top, text="Preorden: -", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                                     font=("Consolas", 9), anchor="w")
        self.lbl_preorden.pack(side="left", padx=(16, 0), fill="x", expand=True)

        self.lbl_estado_eval = tk.Label(
            info_top,
            text="💡 Presiona '📊 Evaluar con CSV' para calcular el error (RMSE, MSE, NUSE, R², ÿ)",
            bg=COLOR_TARJETA, fg=COLOR_SUBTITULO, font=("Segoe UI", 8, "italic")
        )
        self.lbl_estado_eval.pack(side="right")

        # Panel de métricas (solo visible tras evaluar)
        self.frame_metricas_container = tk.Frame(self.card_info, bg=COLOR_TARJETA)

        hdr_met = tk.Frame(self.frame_metricas_container, bg=COLOR_TARJETA)
        hdr_met.pack(fill="x", pady=(4, 2))

        self.lbl_titulo_metricas = tk.Label(
            hdr_met, text="📊 Evaluación de Error contra Dataset CSV:",
            bg=COLOR_TARJETA, fg=COLOR_TITULO, font=("Segoe UI", 9, "bold")
        )
        self.lbl_titulo_metricas.pack(side="left")

        btn_cerrar = tk.Button(hdr_met, text="✕ Ocultar evaluación", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                               activebackground="#faebf2", font=("Segoe UI", 8), relief="flat", cursor="hand2",
                               command=self.ocultar_evaluacion)
        btn_cerrar.pack(side="right")

        self.frame_cajas = tk.Frame(self.frame_metricas_container, bg=COLOR_TARJETA)
        self.frame_cajas.pack(fill="x")

        self.widget_metricas = {}
        metricas_def = [
            ("ÿ (Media de y)", "lbl_y_barra"),
            ("MSE",            "lbl_mse"),
            ("RMSE",           "lbl_rmse"),
            ("NUSE / NMSE",    "lbl_nmse"),
            ("R² (Ajuste)",    "lbl_r2"),
            ("Puntos CSV",     "lbl_puntos"),
        ]
        for nombre, clave in metricas_def:
            caja = tk.Frame(self.frame_cajas, bg=COLOR_METRICA_BG, padx=8, pady=3,
                            highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
            caja.pack(side="left", fill="both", expand=True, padx=2)
            tk.Label(caja, text=nombre, bg=COLOR_METRICA_BG, fg=COLOR_SUBTITULO,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            lbl_val = tk.Label(caja, text="---", bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                               font=("Consolas", 10, "bold"))
            lbl_val.pack(anchor="w")
            self.widget_metricas[clave] = lbl_val

        # Canvas Matplotlib con 2 subplots
        frame_canvas = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_canvas.pack(fill="both", expand=True, pady=(2, 6))

        self.fig = Figure(figsize=(12, 5.2), facecolor=BLANCO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # =========================================================================
    # CONSTRUCCIÓN PESTAÑA 2: ALGORITMO GENÉTICO (REGRESIÓN SIMBÓLICA)
    # =========================================================================
    def _construir_pestana_genetico(self):
        # 1. Barra de configuración
        cfg_panel = tk.Frame(self.tab_genetico, bg=COLOR_TARJETA, padx=14, pady=8,
                             highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        cfg_panel.pack(fill="x", pady=(6, 4))

        tk.Label(cfg_panel, text="Población:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pop = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pop.insert(0, "100")
        self.ent_ga_pop.pack(side="left", padx=(4, 12))

        tk.Label(cfg_panel, text="Generaciones:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_gen = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_gen.insert(0, "100")
        self.ent_ga_gen.pack(side="left", padx=(4, 12))

        tk.Label(cfg_panel, text="Selección:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.cbo_ga_sel = ttk.Combobox(cfg_panel, values=["Ruleta", "Torneo"],
                                       width=8, state="readonly", font=("Segoe UI", 9))
        self.cbo_ga_sel.set("Ruleta")
        self.cbo_ga_sel.pack(side="left", padx=(4, 12))

        tk.Label(cfg_panel, text="Cruza:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pcx = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pcx.insert(0, "0.90")
        self.ent_ga_pcx.pack(side="left", padx=(4, 12))

        tk.Label(cfg_panel, text="Mutación (1%):", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pmut = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                    bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pmut.insert(0, "0.01")
        self.ent_ga_pmut.pack(side="left", padx=(4, 16))

        # Botones de control de evolución
        btn_start = dict(bg=COLOR_BTN_PRIMARIO, fg=BLANCO, activebackground=COLOR_BTN_PRIMARIO_HOVER,
                         font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=12, pady=3)
        btn_sec = dict(bg=COLOR_BTN_SECUNDARIO, fg=COLOR_BTN_SECUNDARIO_TX, activebackground="#f3d3e2",
                       font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                       highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1, padx=10, pady=3)

        self.btn_ga_iniciar = tk.Button(cfg_panel, text="▶ Iniciar Evolución",
                                        command=self.ga_iniciar_o_pausar, **btn_start)
        self.btn_ga_iniciar.pack(side="left", padx=3)

        self.btn_ga_detener = tk.Button(cfg_panel, text="⏹ Detener", command=self.ga_detener,
                                        state="disabled", **btn_sec)
        self.btn_ga_detener.pack(side="left", padx=3)

        self.btn_ga_reiniciar = tk.Button(cfg_panel, text="🔄 Reiniciar Población",
                                          command=self.ga_reiniciar_poblacion, **btn_sec)
        self.btn_ga_reiniciar.pack(side="left", padx=3)

        # Botón para ver el árbol final
        self.btn_ver_arbol_final = tk.Button(
            cfg_panel, text="🌳 Ver Árbol Final", command=self.ga_ver_arbol_final,
            bg="#5a4275", fg=BLANCO, activebackground="#4a3265", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", padx=12, pady=3
        )
        self.btn_ver_arbol_final.pack(side="right", padx=4)

        btn_enviar = tk.Button(
            cfg_panel, text="📋 Enviar a Analizador", command=self.ga_enviar_a_analizador,
            bg=COLOR_TARJETA, fg=COLOR_BTN_PRIMARIO, font=("Segoe UI", 8, "bold"),
            relief="flat", cursor="hand2", highlightbackground=COLOR_BTN_PRIMARIO,
            highlightthickness=1, padx=8, pady=3
        )
        btn_enviar.pack(side="right", padx=4)

        # 2. Card de Métricas en vivo del AG
        card_ga_status = tk.Frame(self.tab_genetico, bg=COLOR_TARJETA, padx=14, pady=6,
                                  highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        card_ga_status.pack(fill="x", pady=4)

        f_status_top = tk.Frame(card_ga_status, bg=COLOR_TARJETA)
        f_status_top.pack(fill="x", pady=(0, 4))

        self.lbl_ga_gen = tk.Label(f_status_top, text="Generación: 0 / 100", bg=COLOR_TARJETA,
                                   fg=COLOR_TITULO, font=("Segoe UI", 10, "bold"))
        self.lbl_ga_gen.pack(side="left")

        self.lbl_ga_expr = tk.Label(f_status_top, text="Mejor Expresión: ---", bg=COLOR_TARJETA,
                                    fg=COLOR_BTN_PRIMARIO, font=("Consolas", 10, "bold"), anchor="w")
        self.lbl_ga_expr.pack(side="left", padx=(20, 0), fill="x", expand=True)

        f_status_met = tk.Frame(card_ga_status, bg=COLOR_TARJETA)
        f_status_met.pack(fill="x")

        self.ga_metricas_labels = {}
        items_ga_met = [
            ("Adap (RMSE)", "lbl_ga_rmse"),
            ("MSE",          "lbl_ga_mse"),
            ("NMSE / NUSE",  "lbl_ga_nmse"),
            ("R² (Ajuste)",  "lbl_ga_r2"),
            ("Estado",       "lbl_ga_estado"),
        ]
        for nom, clv in items_ga_met:
            caja = tk.Frame(f_status_met, bg=COLOR_METRICA_BG, padx=8, pady=3,
                            highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
            caja.pack(side="left", fill="both", expand=True, padx=2)
            tk.Label(caja, text=nom, bg=COLOR_METRICA_BG, fg=COLOR_SUBTITULO,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            lbl_v = tk.Label(caja, text="---", bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                             font=("Consolas", 10, "bold"))
            lbl_v.pack(anchor="w")
            self.ga_metricas_labels[clv] = lbl_v

        # 3. Área dividida: Gráfica de sincronización en vivo + Registro de mutaciones
        split_frame = tk.Frame(self.tab_genetico, bg=COLOR_FONDO_APP)
        split_frame.pack(fill="both", expand=True, pady=(2, 6))

        # Panel izquierdo: Gráfica en vivo
        frame_plot_ga = tk.Frame(split_frame, bg=COLOR_TARJETA,
                                 highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_plot_ga.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.fig_ga = Figure(figsize=(7.5, 4.8), facecolor=BLANCO)
        self.ax_ga = self.fig_ga.add_subplot(1, 1, 1)
        self.canvas_ga = FigureCanvasTkAgg(self.fig_ga, master=frame_plot_ga)
        self.canvas_ga.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

        # Panel derecho: Registro de eventos y mutaciones
        frame_log_ga = tk.Frame(split_frame, bg=COLOR_TARJETA, width=380,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_log_ga.pack(side="right", fill="both", padx=(4, 0))
        frame_log_ga.pack_propagate(False)

        tk.Label(frame_log_ga, text="📝 Registro de Eventos y Mutaciones (1%):",
                 bg=COLOR_TARJETA, fg=COLOR_TITULO, font=("Segoe UI", 9, "bold"), pady=4).pack(anchor="w", padx=6)

        # Scrollbar y Text para el log
        log_scroll = tk.Scrollbar(frame_log_ga)
        log_scroll.pack(side="right", fill="y")

        self.txt_ga_log = tk.Text(frame_log_ga, font=("Consolas", 8), bg="#fffdfb", fg=NEGRO,
                                  yscrollcommand=log_scroll.set, wrap="word", relief="flat")
        self.txt_ga_log.pack(fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        log_scroll.config(command=self.txt_ga_log.yview)

    # =========================================================================
    # LÓGICA DEL ANALIZADOR DE ÁRBOL (PESTAÑA 1)
    # =========================================================================
    def _intervalo(self):
        txt = self.entrada_int.get().strip()
        partes = txt.split(",")
        if len(partes) != 2:
            raise ValueError("El intervalo debe tener el formato a, b (ejemplo: -10, 10)")
        a, b = float(partes[0]), float(partes[1])
        if a >= b:
            raise ValueError("En el intervalo [a, b], 'a' debe ser menor que 'b'")
        return a, b

    def _altura_max(self):
        val = int(self.entrada_alt.get().strip())
        if val < 0 or val > 12:
            raise ValueError("La altura máxima debe estar entre 0 y 12")
        return val

    def cargar_expresion_objetivo(self):
        self.entrada_expr.delete(0, "end")
        self.entrada_expr.insert(0, EXPRESION_OBJETIVO)
        self.entrada_int.delete(0, "end")
        self.entrada_int.insert(0, "-10, 10")
        self.ocultar_evaluacion()
        self.graficar()

    def seleccionar_csv(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV con datos (x, y)",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            try:
                x, y = cargar_csv(ruta)
                self.ruta_csv = ruta
                self.x_csv, self.y_csv = x, y
                messagebox.showinfo("CSV cargado",
                                    f"Se cargaron exitosamente {len(x):,} puntos desde:\n{os.path.basename(ruta)}")
                if self.evaluado_con_csv:
                    self.evaluar_con_csv()
            except Exception as e:
                messagebox.showerror("Error al cargar CSV", f"No se pudo leer el archivo:\n{e}")

    def ocultar_evaluacion(self):
        self.evaluado_con_csv = False
        self.frame_metricas_container.pack_forget()
        self.lbl_estado_eval.pack(side="right")
        if self.arbol_actual:
            self.mostrar(self.arbol_actual)

    def graficar(self):
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión matemática")
            self.arbol_actual = ArbolExpresion(expr)
            self.evaluado_con_csv = False
            self.frame_metricas_container.pack_forget()
            self.lbl_estado_eval.pack(side="right")
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error de Expresión", f"Revisa tus datos:\n{error}")

    def aleatoria(self):
        try:
            a, b = self._intervalo()
            self.arbol_actual = generar_graficable(self._altura_max(), a, b)
            self.entrada_expr.delete(0, "end")
            self.entrada_expr.insert(0, self.arbol_actual.expresion)
            self.evaluado_con_csv = False
            self.frame_metricas_container.pack_forget()
            self.lbl_estado_eval.pack(side="right")
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error", f"Revisa tus datos:\n{error}")

    def evaluar_con_csv(self):
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión matemática antes de evaluar")
            self.arbol_actual = ArbolExpresion(expr)

            if self.x_csv is None or self.y_csv is None:
                self._cargar_datos_csv()
                if self.x_csv is None:
                    raise ValueError("No se pudo cargar el archivo CSV para evaluar")

            _, met = evaluar_arbol_contra_datos(self.arbol_actual, self.x_csv, self.y_csv)

            if met["valido"]:
                self.widget_metricas["lbl_y_barra"].config(text=f"{met['y_barra']:.4f}")
                self.widget_metricas["lbl_mse"].config(text=f"{met['mse']:.6f}")
                self.widget_metricas["lbl_rmse"].config(text=f"{met['rmse']:.6f}")
                self.widget_metricas["lbl_nmse"].config(text=f"{met['nmse']:.6f}")
                self.widget_metricas["lbl_r2"].config(text=f"{met['r2']:.4f}")
                self.widget_metricas["lbl_puntos"].config(text=f"{met['n_validos']:,} / {met['n_total']:,}")
            else:
                for key in self.widget_metricas:
                    self.widget_metricas[key].config(text="Indefinido")

            nom_csv = os.path.basename(self.ruta_csv)
            self.lbl_titulo_metricas.config(
                text=f"📊 Evaluación de Error contra '{nom_csv}' ({len(self.x_csv):,} puntos):"
            )

            self.lbl_estado_eval.pack_forget()
            self.frame_metricas_container.pack(fill="x", pady=(4, 0))

            self.evaluado_con_csv = True
            self.mostrar(self.arbol_actual)

        except Exception as error:
            messagebox.showerror("Error al Evaluar con CSV", str(error))

    def abrir_vista_ampliada(self):
        try:
            a, b = self._intervalo()
            expr = self.entrada_expr.get().strip()
            arbol = ArbolExpresion(expr) if expr else self.arbol_actual
            VentanaVistaAmpliada(self.ventana, arbol, self.x_csv, self.y_csv, a, b)
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo abrir la vista ampliada:\n{error}")

    def mostrar(self, arbol):
        a, b = self._intervalo()
        h, h_max = altura(arbol.raiz), self._altura_max()

        self.fig.clear()
        ax_arbol = self.fig.add_subplot(1, 2, 1)
        ax_func = self.fig.add_subplot(1, 2, 2)

        dibujar_arbol(ax_arbol, arbol)

        datos_ref_plot = None
        if self.evaluado_con_csv and self.x_csv is not None and self.y_csv is not None:
            datos_ref_plot = (self.x_csv, self.y_csv)

        dibujar_funcion(ax_func, arbol, a, b, datos_ref=datos_ref_plot)

        self.fig.tight_layout()
        self.canvas.draw()

        aviso = "  ⚠️ (excede altura máx)" if h > h_max else ""
        self.lbl_altura.config(text=f"Altura: {h} / {h_max}{aviso}")
        tokens_pre = "  ".join(arbol.preorden())
        if len(tokens_pre) > 95:
            tokens_pre = tokens_pre[:92] + "..."
        self.lbl_preorden.config(text=f"Preorden: {tokens_pre}")

    # =========================================================================
    # LÓGICA DEL ALGORITMO GENÉTICO (PESTAÑA 2)
    # =========================================================================
    def ga_preparar(self):
        """Inicializa la instancia del AlgoritmoGenetico si aún no existe."""
        if self.x_csv is None or self.y_csv is None:
            self._cargar_datos_csv()
            if self.x_csv is None:
                raise ValueError("No se encontraron datos CSV para ejecutar la evolución")

        tam_pop = int(self.ent_ga_pop.get().strip())
        self.max_generaciones = int(self.ent_ga_gen.get().strip())
        met_sel = self.cbo_ga_sel.get().strip().lower()
        pcx = float(self.ent_ga_pcx.get().strip())
        pmut = float(self.ent_ga_pmut.get().strip())

        # Para que la evolución sea fluida y responsiva en tiempo real,
        # usamos una resolución adecuada de puntos de evaluación
        paso_muestreo = max(1, len(self.x_csv) // 1500)
        x_eval = self.x_csv[::paso_muestreo]
        y_eval = self.y_csv[::paso_muestreo]

        self.ga = AlgoritmoGenetico(
            x_eval, y_eval, tam_poblacion=tam_pop,
            metodo_seleccion=met_sel, prob_cruza=pcx, prob_mutacion=pmut
        )
        self.txt_ga_log.delete("1.0", "end")
        self._ga_log("Población inicial generada (100 ecuaciones aleatorias).")
        self.ga_actualizar_ui()

    def _ga_log(self, mensaje):
        self.txt_ga_log.insert("end", mensaje + "\n")
        self.txt_ga_log.see("end")

    def ga_iniciar_o_pausar(self):
        if not self.ga_corriendo:
            # Iniciar o reanudar
            try:
                if self.ga is None:
                    self.ga_preparar()
                self.ga_corriendo = True
                self.btn_ga_iniciar.config(text="⏸ Pausar Evolución", bg="#e08e45")
                self.btn_ga_detener.config(state="normal")
                self.ga_metricas_labels["lbl_ga_estado"].config(text="Evolucionando...", fg=COLOR_BTN_PRIMARIO)
                self.ga_iterar()
            except Exception as e:
                messagebox.showerror("Error AG", str(e))
        else:
            # Pausar
            self.ga_pausar()

    def ga_pausar(self):
        self.ga_corriendo = False
        if self.ga_timer_id:
            self.ventana.after_cancel(self.ga_timer_id)
            self.ga_timer_id = None
        self.btn_ga_iniciar.config(text="▶ Reanudar Evolución", bg=COLOR_BTN_PRIMARIO)
        self.ga_metricas_labels["lbl_ga_estado"].config(text="Pausado", fg=COLOR_SUBTITULO)

    def ga_detener(self):
        self.ga_pausar()
        self.btn_ga_iniciar.config(text="▶ Iniciar Evolución", bg=COLOR_BTN_PRIMARIO)
        self.btn_ga_detener.config(state="disabled")
        self.ga_metricas_labels["lbl_ga_estado"].config(text="Detenido", fg=COLOR_SUBTITULO)

    def ga_reiniciar_poblacion(self):
        self.ga_detener()
        self.ga = None
        self.ga_preparar()

    def ga_iterar(self):
        if not self.ga_corriendo or self.ga is None:
            return

        if self.ga.generacion >= self.max_generaciones:
            self.ga_detener()
            self._ga_log(f"🎉 Se completaron las {self.max_generaciones} generaciones!")
            self.ga_metricas_labels["lbl_ga_estado"].config(text="Completado", fg="#2a804d")
            return

        # Ejecutar 1 paso de generación
        res = self.ga.paso_generacion()

        # Reportar mutaciones si ocurrieron en este paso (requisito: decir cuál fue en el 1%)
        for mut_detalle in res["mutaciones_reportadas"]:
            self._ga_log(mut_detalle)

        # Actualizar UI periódicamente
        self.ga_actualizar_ui()

        # Programar siguiente iteración de forma no bloqueante
        self.ga_timer_id = self.ventana.after(15, self.ga_iterar)

    def ga_actualizar_ui(self):
        if self.ga is None or self.ga.mejor_individuo is None:
            return

        ind = self.ga.mejor_individuo
        met = ind.metricas

        self.lbl_ga_gen.config(text=f"Generación: {self.ga.generacion} / {self.max_generaciones}")
        expr_corta = ind.expresion
        if len(expr_corta) > 75:
            expr_corta = expr_corta[:72] + "..."
        self.lbl_ga_expr.config(text=f"Mejor: {expr_corta}")

        self.ga_metricas_labels["lbl_ga_rmse"].config(text=f"{ind.adap:.6f}")
        self.ga_metricas_labels["lbl_ga_mse"].config(text=f"{met.get('mse', 0):.6f}")
        self.ga_metricas_labels["lbl_ga_nmse"].config(text=f"{met.get('nmse', 0):.6f}")
        self.ga_metricas_labels["lbl_ga_r2"].config(text=f"{met.get('r2', 0):.4f}")

        # Dibujar gráfica de sincronización en vivo
        self.ax_ga.clear()
        self.ax_ga.set_facecolor(BLANCO)

        # 1. Curva real del CSV
        if self.x_csv is not None and self.y_csv is not None:
            # Submuestreo para graficación fluida
            paso = max(1, len(self.x_csv) // 800)
            self.ax_ga.plot(self.x_csv[::paso], self.y_csv[::paso], color=AZUL_OBJETIVO,
                            lw=2.0, ls="--", label="Objetivo y (CSV)", alpha=0.85)

        # 2. Curva del mejor individuo actual
        if ind.gen is not None:
            x_plot = np.linspace(-10, 10, 800)
            with np.errstate(all="ignore"):
                y_pred = np.asarray(ind.gen.evaluar(x_plot), dtype=float) * np.ones_like(x_plot)
            y_pred[np.abs(y_pred) > 1e5] = np.nan
            self.ax_ga.plot(x_plot, y_pred, color=ROSA, lw=2.4,
                            label=f"Mejor f(x) (Gen {self.ga.generacion})")

        self.ax_ga.axhline(0, color="#bfa2b3", lw=0.8, ls="-")
        self.ax_ga.axvline(0, color="#bfa2b3", lw=0.8, ls="-")
        self.ax_ga.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.1)

        for spine in self.ax_ga.spines.values():
            spine.set_color("#e3c9d6")

        self.ax_ga.tick_params(colors=TEXTO_MUTED, labelsize=9)
        self.ax_ga.set_title(
            f"Sincronización con CSV (Gen {self.ga.generacion}) • RMSE: {ind.adap:.4f} • R²: {met.get('r2', 0):.4f}",
            color=NEGRO, fontsize=11, fontweight="bold", pad=8
        )
        self.ax_ga.set_xlabel("x", color=NEGRO, fontsize=9, fontweight="bold")
        self.ax_ga.set_ylabel("y", color=NEGRO, fontsize=9, fontweight="bold")
        self.ax_ga.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=8, loc="best")

        self.fig_ga.tight_layout()
        self.canvas_ga.draw()

    def ga_ver_arbol_final(self):
        """Muestra el árbol de expresión final en una ventana dedicada con botón."""
        if self.ga is None or self.ga.mejor_individuo is None:
            messagebox.showinfo("Árbol Final", "Primero inicializa o corre la evolución para tener un árbol final.")
            return

        ind = self.ga.mejor_individuo
        subtitulo = f"(Gen {self.ga.generacion} • RMSE: {ind.adap:.4f})"
        VentanaArbolFinal(self.ventana, ind.gen, titulo_extra=subtitulo)

    def ga_enviar_a_analizador(self):
        """Copia la mejor expresión encontrada a la pestaña principal del analizador."""
        if self.ga is None or self.ga.mejor_individuo is None:
            messagebox.showinfo("Aviso", "Aún no hay una expresión evolucionada.")
            return

        expr = self.ga.mejor_individuo.expresion
        self.entrada_expr.delete(0, "end")
        self.entrada_expr.insert(0, expr)
        self.notebook.select(self.tab_analizador)
        self.evaluado_con_csv = True
        self.evaluar_con_csv()
        messagebox.showinfo(
            "Expresión Transferida",
            f"La expresión evolucionada fue enviada al analizador principal:\n\n{expr}"
        )


if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
