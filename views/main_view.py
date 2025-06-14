from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import time
from models.backpropagation import RedBP
from utils.ui_components import *

class MainView:
    def __init__(self, root):
        # Crear el marco principal
        self.root = root
        self.main_frame = tk.Frame(root, bg=COLOR_BG)
        self.main_frame.pack(fill='both', expand=True)
        
        self.pesos_archivo = None
        self.red = None
        self.datos_entrenamiento = None
        self.datos_salida = None
        self.pesos_cargados = False
        self.suggestions = {}
        self.datos_prueba = None
        self.current_test_case = 0
        self.canvas_error = None
        self.fig_error = None
        self.img_tk = None  # Para mantener referencia a la imagen
        self.entrenamiento_en_progreso = False  # Flag para controlar el entrenamiento

        self.create_main_interface()
        self.style = self.setup_styles()
        self.create_dynamic_config()
        self.add_test_controls()

    def create_dynamic_config(self):
        """Panel de configuración avanzada reorganizado y optimizado"""
        # Panel principal dividido en dos columnas
        main_panel = ttk.Frame(self.config_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_panel.configure(style='TFrame')
        
        # Panel izquierdo - Configuración
        left_panel = ttk.Frame(main_panel)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=10)
        left_panel.configure(style='TFrame')
        
        # Panel derecho - Gráfica de error
        right_panel = ttk.Frame(main_panel)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_panel.configure(style='TFrame')
        
        # ===== PANEL DE CONFIGURACIÓN =====
        config_frame = ttk.LabelFrame(left_panel, text="Configuración de la Red")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        config_frame.configure(borderwidth=2, relief="solid")
        config_frame['style'] = 'Green.TLabelframe'

        # ========== ARQUITECTURA DE LA RED ==========
        arch_frame = ttk.LabelFrame(config_frame, text="Arquitectura de la Red")
        arch_frame.pack(fill=tk.X, padx=5, pady=5)
        arch_frame.configure(borderwidth=2, relief="solid")
        arch_frame['style'] = 'Green.TLabelframe'
        
        # Fila 1 - Capa de Entrada
        entrada_frame = ttk.Frame(arch_frame)
        entrada_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(entrada_frame, text="Entrada:").pack(side=tk.LEFT, padx=2)
        self.entrada_input = ttk.Entry(entrada_frame, width=8)
        self.entrada_input.pack(side=tk.LEFT, padx=2)
        self.lbl_entrada1 = ttk.Label(entrada_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_entrada1.pack(side=tk.LEFT, padx=2)

        # Fila 2 - Capa Oculta
        oculta_frame = ttk.Frame(arch_frame)
        oculta_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(oculta_frame, text="Oculta:").pack(side=tk.LEFT, padx=2)
        self.oculta_input = ttk.Entry(oculta_frame, width=8)
        self.oculta_input.pack(side=tk.LEFT, padx=2)
        self.lbl_oculta1 = ttk.Label(oculta_frame, text="Sugerido", font=("Arial", 9, "italic"))
        self.lbl_oculta1.pack(side=tk.LEFT, padx=2)

        # Fila 3 - Capa de Salida
        salida_frame = ttk.Frame(arch_frame)
        salida_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(salida_frame, text="Salida:").pack(side=tk.LEFT, padx=2)
        self.salida_input = ttk.Entry(salida_frame, width=8)
        self.salida_input.pack(side=tk.LEFT, padx=2)
        self.lbl_salida1 = ttk.Label(salida_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_salida1.pack(side=tk.LEFT, padx=2)

        # ========== PARÁMETROS DE ENTRENAMIENTO ==========
        param_frame = ttk.LabelFrame(config_frame, text="Parámetros de Entrenamiento")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        param_frame.configure(borderwidth=2, relief="solid")
        param_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en dos columnas
        param_grid = ttk.Frame(param_frame)
        param_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(param_grid, text="Alpha (α):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.alpha_input = ttk.Entry(param_grid, width=8)
        self.alpha_input.insert(0, "0.01")
        self.alpha_input.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Precisión:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.precision_input = ttk.Entry(param_grid, width=8)
        self.precision_input.insert(0, "0.01")
        self.precision_input.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Momentum:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.momentum_check = ttk.Checkbutton(param_grid, command=self.toggle_momentum)
        self.momentum_check.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Columna 2
        ttk.Label(param_grid, text="Máx. Épocas:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.max_epocas_input = ttk.Entry(param_grid, width=8)
        self.max_epocas_input.insert(0, "1000000")
        self.max_epocas_input.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Bias:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.bias_input = ttk.Entry(param_grid, width=8)
        self.bias_input.insert(0, "1.0")
        self.bias_input.grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Beta (β):").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        self.beta_input = ttk.Entry(param_grid, width=8, state='disabled')
        self.beta_input.grid(row=2, column=3, sticky="w", padx=5, pady=2)

        # ========== FUNCIONES DE ACTIVACIÓN ==========
        activ_frame = ttk.LabelFrame(config_frame, text="Funciones de Activación")
        activ_frame.pack(fill=tk.X, padx=5, pady=5)
        activ_frame.configure(borderwidth=2, relief="solid")
        activ_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en tres columnas
        activ_grid = ttk.Frame(activ_frame)
        activ_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(activ_grid, text="Capa Oculta:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.func_oculta = ttk.Combobox(activ_grid, 
                                    values=['Sigmoide', 'ReLU', 'Tanh', 'Leaky ReLU'], 
                                    state='readonly',
                                    width=12)
        self.func_oculta.current(0)
        self.func_oculta.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.func_oculta.bind("<<ComboboxSelected>>", self.actualizar_interfaz_activacion)
        
        # Columna 2
        ttk.Label(activ_grid, text="Capa Salida:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.func_salida = ttk.Combobox(activ_grid, 
                                    values=['Sigmoide', 'Lineal', 'Softmax'], 
                                    state='readonly',
                                    width=12)
        self.func_salida.current(0)
        self.func_salida.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        # Columna 3
        ttk.Label(activ_grid, text="Leaky ReLU:").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        self.beta_oculta_input = ttk.Entry(activ_grid, width=8)
        self.beta_oculta_input.insert(0, "0.01")
        self.beta_oculta_input.grid(row=0, column=5, sticky="w", padx=5, pady=2)

        # ===== PANEL DE DATOS Y ACCIONES =====
        data_frame = ttk.LabelFrame(config_frame, text="Datos de Entrenamiento")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        data_frame.configure(borderwidth=2, relief="solid")
        data_frame['style'] = 'Green.TLabelframe'
        
        # Botones de carga de datos y entrenamiento
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        btn_cargar_entrada = ModernButton(
            btn_frame, 
            text="Cargar Entradas", 
            command=self.cargar_archivo_entrada,
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        btn_cargar_entrada.pack(side=tk.LEFT, padx=5)
        
        btn_cargar_salida = ModernButton(
            btn_frame, 
            text="Cargar Salidas", 
            command=self.cargar_archivo_salida,
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        btn_cargar_salida.pack(side=tk.LEFT, padx=5)
        
        # Botón de entrenamiento
        btn_frame2 = ttk.Frame(data_frame)
        btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_entrenar = ModernButton(
            btn_frame2, 
            text="Entrenar Red", 
            command=self.entrenar_red,
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        self.btn_entrenar.pack(side=tk.LEFT, padx=5)
        
        # Barra de progreso para el entrenamiento
        progress_frame = ttk.Frame(data_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progreso:").pack(side=tk.LEFT, padx=2)
        self.progress_bar = ttk.Progressbar(progress_frame, style="Green.Horizontal.TProgressbar", length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=2)

        # ===== PANEL DE RESULTADOS DEL ENTRENAMIENTO =====
        results_frame = ttk.LabelFrame(right_panel, text="Resultados del Entrenamiento")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_frame.configure(borderwidth=2, relief="solid")
        results_frame['style'] = 'Green.TLabelframe'
        
        # Estado del entrenamiento
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Estado: No entrenado", foreground="red", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Canvas(status_frame, width=15, height=15, bg='white', highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
        
        # Indicadores de métricas
        metrics_frame = ttk.LabelFrame(results_frame, text="Indicadores de Métricas")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        metrics_frame.configure(borderwidth=2, relief="solid")
        metrics_frame['style'] = 'Green.TLabelframe'
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(padx=5, pady=5, side=tk.LEFT)
        
        ttk.Label(metrics_grid, text="Épocas:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epochs_value = ttk.Label(metrics_grid, text="-")
        self.epochs_value.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_grid, text="Error Final:").grid(row=1, column=0, sticky="w", padx=5, pady=2)   
        self.error_value = ttk.Label(metrics_grid, text="-")
        self.error_value.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Gráfica de error
        self.error_graph_frame = ttk.LabelFrame(results_frame, text="Evolución del Error vs Épocas")
        self.error_graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.error_graph_frame.configure(borderwidth=2, relief="solid")
        self.error_graph_frame['style'] = 'Green.TLabelframe'
        
        # Mensaje inicial en la gráfica
        msg_frame = ttk.Frame(self.error_graph_frame)
        msg_frame.pack(expand=True)
        
        ttk.Label(msg_frame, 
                 text="La gráfica de error se mostrará aquí después del entrenamiento",
                 font=("Arial", 10, "italic")).pack(pady=50)

    def actualizar_interfaz_activacion(self, event=None):
        """Actualiza la interfaz según la función de activación seleccionada"""
        func_oculta = self.func_oculta.get().lower()
        
        # Habilitar o deshabilitar el campo beta de Leaky ReLU
        if func_oculta == 'leaky relu':
            self.beta_oculta_input.config(state='normal')
        else:
            self.beta_oculta_input.config(state='disabled')

    def toggle_momentum(self):
        """Habilita o deshabilita el campo beta de momentum"""
        if self.momentum_check.instate(['selected']):
            self.beta_input.config(state='normal')
        else:
            self.beta_input.config(state='disabled')

    def actualizar_sugerencias(self):
        """Actualizar todas las sugerencias de capas"""
        if (self.datos_entrenamiento is not None and 
        self.datos_salida is not None and
        len(self.datos_entrenamiento) > 0 and 
        len(self.datos_salida) > 0):

            # Calcular valores
            entrada = len(self.datos_entrenamiento[0])
            salida = len(self.datos_salida[0])
            oculta_sugerida = int(np.sqrt(entrada * salida))

            # Actualizar labels
            self.lbl_entrada1.config(text=f"Automático ({entrada})")
            self.lbl_oculta1.config(text=f"Sugerido ({oculta_sugerida})")
            self.lbl_salida1.config(text=f"Automático ({salida})")

            # Actualizar campos editables
            self.entrada_input.delete(0, tk.END)
            self.entrada_input.insert(0, str(entrada))
            self.oculta_input.delete(0, tk.END)
            self.oculta_input.insert(0, str(oculta_sugerida))
            self.salida_input.delete(0, tk.END)
            self.salida_input.insert(0, str(salida))

    def get_config(self):
        """Obtiene la configuración de la red desde la interfaz"""
        try:
            # Obtener valores de la arquitectura
            capa_entrada = int(self.entrada_input.get())
            capa_oculta = int(self.oculta_input.get())
            capa_salida = int(self.salida_input.get())
            
            # Obtener valores de los parámetros de entrenamiento
            alfa = float(self.alpha_input.get())
            max_epocas = int(self.max_epocas_input.get())
            precision = float(self.precision_input.get())
            
            # Obtener valor de bias
            bias = float(self.bias_input.get()) > 0
            
            # Obtener funciones de activación
            func_oculta = self.func_oculta.get().lower()
            func_salida = self.func_salida.get().lower()
            
            # Obtener beta para Leaky ReLU si es necesario
            beta_leaky_relu = float(self.beta_oculta_input.get())
            
            # Obtener valor de momentum si está habilitado
            momentum = False
            beta = 0.0
            if self.momentum_check.instate(['selected']):
                momentum = True
                beta = float(self.beta_input.get())
        
            # Crear diccionario de configuración
            config = {
                'capa_entrada': capa_entrada,
                'capa_oculta': capa_oculta,
                'capa_salida': capa_salida,
                'alfa': alfa,
                'max_epocas': max_epocas,
                'precision': precision,
                'bias': bias,
                'funciones_activacion': [func_oculta, func_salida],
                'beta_leaky_relu': beta_leaky_relu,
                'momentum': momentum,
                'beta': beta
            }
        
            return config
    
        except ValueError as e:
            print(f"Error en la configuración: {str(e)}")
            print("Verifique que todos los campos numéricos contengan valores válidos")
            return None
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            return None

    def create_main_interface(self):
        """Crea la interfaz principal con pestañas"""
        # Crear encabezado
        self.create_header()

        # Crear un contenedor para el contenido principal
        content_container = tk.Frame(self.main_frame, bg=COLOR_BG)
        content_container.pack(fill='both', expand=True, padx=5, pady=10)

        # Crear un canvas con scrollbar para permitir desplazamiento
        self.canvas = tk.Canvas(content_container, bg=COLOR_BG, highlightthickness=0)
        
        # Crear scrollbar moderna
        self.scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        
        self.canvas.pack(side=tk.LEFT, fill='both', expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Crear frame para el contenido scrollable
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLOR_BG)
        self.scrollable_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configurar eventos de scroll
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Usar bind_all con un tag único para evitar conflictos
        self.mousewheel_tag = "MainView_MouseWheel"
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        
        # Crear pestañas con estilo mejorado
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=10)
        self.notebook.configure(style='TNotebook')
    
        # Pestaña para configuración y entrenamiento
        self.config_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.config_frame, text="Configuración y Entrenamiento")
        
        # Pestaña para pruebas personalizadas
        self.test_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.test_frame, text="Pruebas Personalizadas")
        
        # Crear pie de página con copyright siempre visible
        self.create_footer()
    
    def _bind_mousewheel(self, event):
        """Vincula el evento de la rueda del mouse cuando el cursor entra en el canvas"""
        self.root.bind_all(f"<MouseWheel>", self.on_mousewheel, add="+")
        
    def _unbind_mousewheel(self, event):
        """Desvincula el evento de la rueda del mouse cuando el cursor sale del canvas"""
        self.root.unbind_all(f"<MouseWheel>")
    
    def on_frame_configure(self, event=None):
        """Configura el canvas para scrollear todo el contenido"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event=None):
        """Ajusta el ancho del frame scrollable al canvas"""
        if event and event.width > 0:
            canvas_width = event.width
            self.canvas.itemconfig(self.scrollable_window, width=canvas_width)

    def on_mousewheel(self, event=None):
        """Maneja el evento de la rueda del mouse para scrollear"""
        if event:
            # Normalizar el delta para diferentes sistemas
            delta = event.delta if hasattr(event, 'delta') else -1 * event.num if event.num in (4, 5) else 0
            if delta:
                self.canvas.yview_scroll(int(-1*(delta/120)), "units")

    def setup_styles(self):
        """Configurar estilos para ttk widgets con apariencia más profesional"""
        style = ttk.Style()
        style.theme_use('default')
        
        # Estilo para pestañas
        style.configure('TNotebook', background=COLOR_BG)
        style.configure('TNotebook.Tab', background=COLOR_BG, foreground=COLOR_TEXT, 
                        padding=[15, 5], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', 
                background=[('selected', COLOR_PRIMARY)], 
                foreground=[('selected', 'white')])
        
        # Estilo para frames
        style.configure('TFrame', background='white')
        
        # Estilo para labels
        style.configure('TLabel', background='white')
        
        # Estilo para combobox
        style.configure('TCombobox', 
                        fieldbackground='white',
                        background=COLOR_PRIMARY,
                        foreground=COLOR_TEXT,
                        arrowcolor=COLOR_PRIMARY)
        style.map('TCombobox',
                fieldbackground=[('readonly', 'white')],
                selectbackground=[('readonly', COLOR_PRIMARY)],
                selectforeground=[('readonly', 'white')])
        
        # Estilo para separadores
        style.configure('TSeparator', background=COLOR_PRIMARY)
        
        # Estilo para barras de progreso
        style.configure("TProgressbar", 
                        troughcolor='white', 
                        background=COLOR_PRIMARY,
                        thickness=8)
        
        # Estilo para barra de progreso verde
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor='white',
                        background=COLOR_PRIMARY,
                        thickness=10)
        
        # Estilo para scrollbars
        style.configure("TScrollbar",
                        background='white',
                        troughcolor='white',
                        arrowsize=14)
        
        # Estilo para LabelFrame con borde verde
        style.configure('Green.TLabelframe', 
                        bordercolor=COLOR_PRIMARY,
                        background='white')
        style.configure('Green.TLabelframe.Label', 
                        foreground=COLOR_PRIMARY,
                        background='white',
                        font=('Arial', 10, 'bold'))
        
        # Estilo para botones
        style.configure('TButton', 
                        background='white',
                        foreground=COLOR_TEXT)
        
        # Estilo para checkbuttons
        style.configure('TCheckbutton', 
                        background='white')
        
        # Estilo para entradas
        style.configure('TEntry', 
                        fieldbackground='white')
        
        return style
  
    def obtener_ruta_relativa(self, ruta_archivo):
        if getattr(sys, 'frozen', False):  # Si el programa está empaquetado con PyInstaller
            base_path = sys._MEIPASS       # Carpeta temporal donde PyInstaller extrae archivos
        else:
            base_path = os.path.abspath(".")  # Carpeta normal en modo desarrollo

        return os.path.join(base_path, ruta_archivo)

    def add_authors_info(self):
        # Crear un marco para la información de autores con estilo mejorado
        authors_frame = tk.Frame(self.main_frame, bg=COLOR_BG, padx=10, pady=5)  # Reducir padding
        authors_frame.pack(fill=tk.X, before=self.notebook)
        
        # Añadir un borde superior e inferior sutil
        top_border = tk.Frame(authors_frame, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        bottom_border = tk.Frame(authors_frame, bg=COLOR_BORDER, height=1)
        bottom_border.pack(side=tk.BOTTOM, fill='x')
        
        # Información de los autores con mejor tipografía
        authors_info = tk.Label(
            authors_frame,
            text="Desarrollado por: Sergio Leonardo Moscoso Ramirez - Miguel Ángel Pardo Lopez",
            font=("Arial", 10),  # Reducir tamaño
            bg=COLOR_BG,
            fg=COLOR_TEXT
        )
        authors_info.pack(side=tk.LEFT, padx=5)  # Reducir padding
        
    def create_footer(self):
        # Crear un marco para el pie de página con estilo mejorado
        footer_frame = tk.Frame(self.main_frame, bg=COLOR_PRIMARY, height=30)  # Reducir altura
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Añadir un borde superior sutil
        top_border = tk.Frame(footer_frame, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        # Texto del pie de página con mejor tipografía
        footer_text = "© Universidad de Cundinamarca - Simulador de BackPropagation"
        footer_label = tk.Label(footer_frame, text=footer_text, 
                                font=("Arial", 8), bg=COLOR_PRIMARY, fg="white")  # Reducir tamaño
        footer_label.pack(pady=6)  # Reducir padding

    def create_header(self):
        # Crear un marco para el encabezado con borde inferior sutil
        header_frame = tk.Frame(self.main_frame, bg=COLOR_BG, height=0)  # Reducir altura
        header_frame.pack(fill='x', pady=(0, 0))  # Reducir espacio
        
        # Añadir un borde inferior sutil
        border_frame = tk.Frame(header_frame, bg=COLOR_BG, height=1)
        border_frame.pack(side=tk.BOTTOM, fill='x')
        
        try:
            # Tamaños deseados
            logo_with = 60   # Reducir ancho
            logo_height = 80  # Reducir alto

            # Crear un frame para contener la imagen
            logo_frame = tk.Frame(header_frame, width=logo_with, height=logo_height, bg=COLOR_BG)
            logo_frame.pack(side=tk.LEFT, padx=15, pady=5)  # Reducir padding
            
            try:
                # Obtener la ruta de la imagen de manera segura
                image_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
                
                # Cargar y redimensionar la imagen
                image = Image.open(image_path)
                image = image.resize((logo_with, logo_height), Image.LANCZOS)
                logo_img = ImageTk.PhotoImage(image)

                # Crear un Label con la imagen
                logo_label = tk.Label(logo_frame, image=logo_img, bg=COLOR_BG)
                logo_label.image = logo_img  # Mantener referencia para que no se "pierda" la imagen
                logo_label.pack()

            except Exception as e:
                print(f"Error al cargar la imagen: {e}")
                
                # Como respaldo, dibujamos un canvas con un óvalo verde y texto "UDEC"
                logo_canvas = tk.Canvas(
                    logo_frame, 
                    width=logo_with, 
                    height=logo_height, 
                    bg=COLOR_LIGHT_BG, 
                    highlightthickness=0
                )
                logo_canvas.pack()
                
                logo_canvas.create_oval(
                    5, 5, 
                    logo_with - 5, logo_height - 5, 
                    fill="#006633", 
                    outline=""
                )
                logo_canvas.create_text(
                    logo_with / 2, logo_height / 2, 
                    text="UDEC", 
                    fill="white", 
                    font=("Arial", 10, "bold")  # Reducir tamaño
                )

        except Exception as e:
            print(f"Error en la creación del logo: {e}")
            
        # Título y subtítulo con mejor tipografía
        title_frame = tk.Frame(header_frame, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT, padx=8, pady=5)  # Reducir padding
        
        # Información del proyecto con mejor alineación
        info_frame = tk.Frame(header_frame, bg=COLOR_BG)
        info_frame.pack(side=tk.RIGHT, padx=15, pady=5)  # Reducir padding
        
        # Crear un marco para la información de autores con estilo mejorado
        authors_frame = tk.Frame(self.main_frame, bg=COLOR_BG, padx=10, pady=5)  # Reducir padding
        authors_frame.pack(fill=tk.X, padx=20, pady=5)
        
        bottom_border = tk.Frame(authors_frame, bg=COLOR_BORDER, height=1)
        bottom_border.pack(side=tk.BOTTOM, fill='x')
        
        title_label = tk.Label(title_frame, text="BACKPROPAGATION", 
                            font=("Arial", 20, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)  # Reducir tamaño
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(title_frame, text="Universidad de Cundinamarca", 
                                font=("Arial", 12), bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY)  # Reducir tamaño
        subtitle_label.pack(anchor='w')
        
        authors_info = tk.Label(authors_frame, text="Autores: Sergio Leonardo Moscoso Ramirez - Miguel Ángel Pardo Lopez",
                                 font=("Segoe UI", 10, "bold"),
                                 bg=COLOR_BG,
                                 fg=COLOR_TEXT)
        authors_info.pack(side=tk.LEFT, padx=5)

    def add_training_controls(self):
        # Modificar frame de controles
        frame_controles = ttk.Frame(self.config_frame)
        frame_controles.pack(pady=10)
        
        # Botones y controles de entrenamiento
        btn_cargar_entrada = ModernButton(
            frame_controles, 
            text="Cargar Entradas", 
            command=self.cargar_archivo_entrada,
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        btn_cargar_entrada.pack(side=tk.LEFT, padx=5)
        
        btn_cargar_salida = ModernButton(
            frame_controles, 
            text="Cargar Salidas", 
            command=self.cargar_archivo_salida,
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        btn_cargar_salida.pack(side=tk.LEFT, padx=5)
        
        btn_entrenar = ModernButton(
            frame_controles, 
            text="Entrenar Red", 
            command=self.entrenar_red,
            bg=COLOR_SUCCESS, 
            fg='white',
            hover_bg='#218838'
        )
        btn_entrenar.pack(side=tk.LEFT, padx=5)
        
        btn_guardar_pesos = ModernButton(
            frame_controles, 
            text="Guardar Pesos", 
            command=self.guardar_pesos,
            bg=COLOR_ACCENT_BLUE, 
            fg=COLOR_TEXT,
            hover_bg='#3399ff'
        )
        btn_guardar_pesos.pack(side=tk.LEFT, padx=5)
        
        btn_cargar_pesos = ModernButton(
            frame_controles, 
            text="Cargar Pesos", 
            command=self.cargar_pesos,
            bg=COLOR_ACCENT_BLUE, 
            fg=COLOR_TEXT,
            hover_bg='#3399ff'
        )
        btn_cargar_pesos.pack(side=tk.LEFT, padx=5)
        
        # Área de registro
        self.log_text = tk.Text(self.config_frame, height=8, state='disabled')
        self.log_text.pack(fill=tk.X, padx=5, pady=5)

    def add_test_controls(self):
        """Interfaz mejorada para pruebas personalizadas con cuadrícula 5x7"""
        # Marco principal para pruebas con mejor organización
        main_test_frame = ttk.Frame(self.test_frame)
        main_test_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Panel Izquierdo: Entrada y Controles ---
        left_panel = ttk.LabelFrame(main_test_frame, text="Patrón de Entrada")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_panel.configure(borderwidth=2, relief="solid")
        left_panel['style'] = 'Green.TLabelframe'

        # Instrucciones para el usuario
        instructions = ttk.Label(left_panel, 
                               text="Haga clic en las celdas para activar/desactivar o cargue un archivo de prueba en formato txt.",
                               wraplength=250, justify='center', font=("Arial", 9, "italic"))
        instructions.pack(pady=5)

        # Cuadrícula de entrada (5x7) con mejor presentación
        grid_frame = ttk.Frame(left_panel)
        grid_frame.pack(pady=10)
        
        self.input_grid = []
        for i in range(7):  # 7 filas
            row = []
            for j in range(5):  # 5 columnas
                cell = tk.Canvas(grid_frame, width=35, height=35, bg='white', 
                                highlightthickness=1, highlightbackground=COLOR_BORDER)
                cell.grid(row=i, column=j, padx=2, pady=2)
                # Añadir evento de clic para cada celda
                cell.bind("<Button-1>", lambda event, r=i, c=j: self.toggle_cell(r, c))
                row.append(cell)
            self.input_grid.append(row)

        # Botones de control en un frame horizontal
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=10, fill=tk.X)
        
        self.btn_cargar_prueba = ModernButton(btn_frame, text="Cargar Archivo de Prueba", 
                                             command=self.cargar_archivo_prueba,
                                             bg=COLOR_PRIMARY, fg='white')
        self.btn_cargar_prueba.pack(side=tk.LEFT, padx=5)
        
        self.btn_limpiar = ModernButton(btn_frame, text="Limpiar Patrón", 
                                      command=self.limpiar_patron,
                                      bg=COLOR_PRIMARY, fg='white')
        self.btn_limpiar.pack(side=tk.LEFT, padx=5)
        
        self.btn_cargar_pesos = ModernButton(btn_frame, text="Cargar Pesos", 
                                           command=self.cargar_pesos,
                                           bg=COLOR_PRIMARY, fg='white')
        self.btn_cargar_pesos.pack(side=tk.LEFT, padx=5)

        # --- Panel Derecho: Resultados Visuales ---
        right_panel = ttk.Frame(main_test_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Resultado de la clasificación
        result_frame = ttk.LabelFrame(right_panel, text="Resultado de la Clasificación")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        result_frame.configure(borderwidth=2, relief="solid")
        result_frame['style'] = 'Green.TLabelframe'
        
        # Canvas para dibujar la letra reconocida
        canvas_frame = ttk.Frame(result_frame)
        canvas_frame.pack(pady=10, expand=True)
        
        canvas_label = ttk.Label(canvas_frame, text="Vocal Clasificada:", font=("Arial", 9, "italic"))
        canvas_label.pack(anchor='center')
        
        self.canvas_letra = tk.Canvas(canvas_frame, width=200, height=200, bg='white',
                            highlightthickness=1, highlightbackground=COLOR_PRIMARY)
        self.canvas_letra.pack(pady=5)

        # Barras de activación por vocal con mejor presentación
        probs_frame = ttk.LabelFrame(right_panel, text="Nivel de Activación")
        probs_frame.pack(fill=tk.X, pady=10, padx=5)
        probs_frame.configure(borderwidth=2, relief="solid")
        probs_frame['style'] = 'Green.TLabelframe'
        
        self.barras_similitud = {}
        self.lbl_porcentajes = {}
        vocales = ['A', 'E', 'I', 'O', 'U']
        
        for idx, vocal in enumerate(vocales):
            frame = ttk.Frame(probs_frame)
            frame.pack(fill=tk.X, pady=3, padx=5)
            
            ttk.Label(frame, text=f"{vocal}:", width=3).pack(side=tk.LEFT, padx=5)
            
            self.barras_similitud[vocal] = ttk.Progressbar(frame, length=150)
            self.barras_similitud[vocal].pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            
            self.lbl_porcentajes[vocal] = ttk.Label(frame, text="0.0%", width=6)
            self.lbl_porcentajes[vocal].pack(side=tk.RIGHT, padx=5)

    def limpiar_patron(self):
        """Limpia el patrón actual en la cuadrícula"""
        for i in range(len(self.input_grid)):
            for j in range(len(self.input_grid[0])):
                self.input_grid[i][j].config(bg='white')
        
        # Limpiar resultados
        for vocal in ['A', 'E', 'I', 'O', 'U']:
            self.barras_similitud[vocal]['value'] = 0
            self.lbl_porcentajes[vocal].config(text="0.0%")
        
        self.canvas_letra.delete("all")
        self.mostrar_resultados(["Patrón limpiado"])

    def toggle_cell(self, row, col):
        """Cambia el estado de una celda en la cuadrícula y actualiza la visualización"""
        # Obtener el color actual
        current_color = self.input_grid[row][col].cget("bg")
        
        # Cambiar el color
        new_color = COLOR_PRIMARY if current_color == 'white' else 'white'
        self.input_grid[row][col].config(bg=new_color)
        
        # Si hay una red entrenada, clasificar el patrón actual
        if self.red is not None:
            self.probar_patron_actual()

    def obtener_patron_actual(self):
        """Obtiene el patrón actual de la cuadrícula como un vector"""
        patron = []
        for i in range(len(self.input_grid)):
            for j in range(len(self.input_grid[0])):
                valor = 1 if self.input_grid[i][j].cget("bg") == COLOR_PRIMARY else 0
                patron.append(valor)
        return np.array(patron)
    
    def probar_patron_actual(self):
        """Clasifica el patrón actual con la red neuronal"""
        if self.red is None:
            print("Error: Primero debe entrenar la red o cargar pesos")
            return
        
        try:
            # Obtener el patrón actual
            patron = self.obtener_patron_actual()
            
            # Realizar clasificación
            salida = self.red.predecir([patron])[0]
            
            # Calcular porcentajes de activación
            activaciones = {vocal: float(salida[i])*100 for i, vocal in enumerate(['A', 'E', 'I', 'O', 'U'])}
            
            # Actualizar barras de activación
            self.actualizar_barras(activaciones)
            
            # Determinar la vocal clasificada
            vocal_clasificada = max(activaciones, key=activaciones.get)
            nivel_activacion = activaciones[vocal_clasificada]
            
            # Dibujar la letra
            self.dibujar_letra(vocal_clasificada)
            
            # Actualizar resultados en texto con información más detallada
            resultados = [
                f"Clasificación: {vocal_clasificada} ({nivel_activacion:.2f}%)",
                f"Confianza: {'Alta' if nivel_activacion > 80 else 'Media' if nivel_activacion > 50 else 'Baja'}",
                f"Valores de activación: {', '.join([f'{float(v):.4f}' for v in salida])}"
            ]
            self.mostrar_resultados(resultados)
            
        except Exception as e:
            print(f"Error en la clasificación: {str(e)}")

    def dibujar_letra(self, letra):
        """Muestra la imagen de la vocal clasificada"""
        self.canvas_letra.delete("all")
        
        try:
            # Ruta a la imagen de la vocal
            ruta_imagen = f"utils\\Images\\Vocales_Images\\{letra}.png"
            
            # Cargar la imagen
            imagen = Image.open(ruta_imagen)
            
            # Redimensionar la imagen para que se ajuste al canvas (manteniendo la proporción)
            ancho_canvas = 200
            alto_canvas = 200
            
            # Calcular el tamaño para mantener la proporción
            ancho_img, alto_img = imagen.size
            ratio = min(ancho_canvas/ancho_img, alto_canvas/alto_img)
            nuevo_ancho = int(ancho_img * ratio)
            nuevo_alto = int(alto_img * ratio)
            
            imagen = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
            
            # Convertir la imagen para Tkinter
            self.img_tk = ImageTk.PhotoImage(imagen)
            
            # Calcular posición para centrar la imagen
            x = (ancho_canvas - nuevo_ancho) // 2
            y = (alto_canvas - nuevo_alto) // 2
            
            # Mostrar la imagen en el canvas
            self.canvas_letra.create_image(x + nuevo_ancho//2, y + nuevo_alto//2, image=self.img_tk)
            
            # Añadir un título
            self.canvas_letra.create_text(100, 20, text=letra, font=("Arial", 24, "bold"), fill=COLOR_PRIMARY)
            
        except Exception as e:
            # Si hay un error al cargar la imagen, mostrar un mensaje
            self.canvas_letra.create_text(100, 100, text=f"Vocal: {letra}", font=("Arial", 24, "bold"), fill=COLOR_PRIMARY)
            print(f"Error al cargar imagen de vocal: {str(e)}")

    def actualizar_barras(self, activaciones):
        """Actualiza las barras de progreso con los niveles de activación de cada vocal"""
        # Encontrar la vocal con mayor nivel de activación
        max_vocal = max(activaciones, key=activaciones.get)
        
        for vocal, barra in self.barras_similitud.items():
            valor = activaciones.get(vocal, 0)
            barra['value'] = valor
            
            # Formatear el texto del porcentaje
            texto_porcentaje = f"{valor:.1f}%"
            
            # Destacar la vocal con mayor nivel de activación
            if vocal == max_vocal:
                self.lbl_porcentajes[vocal].config(
                    text=texto_porcentaje,
                    font=("Arial", 9, "bold"),
                    foreground=COLOR_PRIMARY
                )
            else:
                self.lbl_porcentajes[vocal].config(
                    text=texto_porcentaje,
                    font=("Arial", 9),
                    foreground=COLOR_TEXT
                )

    def cargar_archivo_prueba(self):
        """Carga un archivo de prueba con patrones 5x7"""
        archivo = filedialog.askopenfilename(filetypes=[("Archivos TXT", "*.txt")])
        if archivo:
            try:
                # Cargar patrones de prueba
                with open(archivo, 'r') as f:
                    self.datos_prueba = [np.array(eval(line)) for line in f]
                
                if len(self.datos_prueba) > 0:
                    # Verificar que los patrones tengan el tamaño correcto (5x7 = 35)
                    if len(self.datos_prueba[0]) != 35:
                        print(f"Error: Los patrones deben ser de 5x7 (35 elementos), pero tienen {len(self.datos_prueba[0])} elementos")
                        self.datos_prueba = None
                        return
                        
                    self.current_test_case = 0
                    self.mostrar_patron(self.datos_prueba[0])
                    print(f"Cargados {len(self.datos_prueba)} patrones de prueba")
                    
                    # Si hay una red entrenada, probar el primer patrón
                    if self.red is not None:
                        self.probar_patron_actual()
                else:
                    print("El archivo no contiene patrones válidos")
                    
            except Exception as e:
                print(f"Error al cargar archivo de prueba: {str(e)}")
                self.datos_prueba = None

    def mostrar_resultados(self, lineas):
        """Muestra resultados en el área de texto"""
        # Implementación para mostrar resultados en la interfaz
        pass

    def cargar_archivo_entrada(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos TXT", "*.txt")])
        if archivo:
            try:
                with open(archivo, 'r') as f:
                    datos = []
                    for line in f:
                        # Convertir a array numpy y verificar consistencia
                        arr = np.array(eval(line.strip()))
                        if not isinstance(arr, np.ndarray):
                            raise ValueError("Formato de entrada inválido")
                        datos.append(arr)
                    
                    # Verificar que todos tengan la misma longitud
                    longitudes = [len(x) for x in datos]
                    if len(set(longitudes)) != 1:
                        print("Error: Entradas con tamaños diferentes")
                        return
                        
                    self.datos_entrenamiento = datos
                    print(f"Cargadas {len(datos)} entradas válidas")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
        self.actualizar_sugerencias()

    def cargar_archivo_salida(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos TXT", "*.txt")])
        if archivo:
            try:
                with open(archivo, 'r') as f:
                    salidas = [line.strip() for line in f]
                    # Convertir a array NumPy explícitamente
                    self.datos_salida = np.array([self.codificar_salida(s) for s in salidas])
                    
                    # Verificar que el array no esté vacío
                    if self.datos_salida.size == 0:
                        raise ValueError("Archivo de salida vacío")
                print(f"Cargadas {self.datos_salida.shape[0]} salidas válidas")
                self.actualizar_sugerencias()
            except Exception as e:
                print(f"Error: {str(e)}")
                self.datos_salida = None

    def codificar_salida(self, letra):
        codigo = {'A': [1,0,0,0,0], 'E': [0,1,0,0,0], 
                 'I': [0,0,1,0,0], 'O': [0,0,0,1,0], 
                 'U': [0,0,0,0,1]}
        return codigo.get(letra.upper(), [0]*5)

    def decodificar_salida(self, vector):
        umbral = 0.01
        vocales = ['A', 'E', 'I', 'O', 'U']
        idx = np.argmax(vector)
        return vocales[idx] if vector[idx] > umbral else "Desconocido"

    def entrenar_red(self):
        """Entrena la red neuronal con los datos cargados y actualiza la interfaz"""
        # Evitar múltiples entrenamientos simultáneos
        if self.entrenamiento_en_progreso:
            print("Ya hay un entrenamiento en curso. Espere a que termine.")
            return
            
        try:
            # Validar que los datos estén cargados
            if self.datos_entrenamiento is None or self.datos_salida is None:
                print("Error: Cargue los datos de entrenamiento y salida primero")
                return

            # Obtener configuración
            config = self.get_config()
            if config is None:
                return
                
            # Marcar inicio del entrenamiento
            self.entrenamiento_en_progreso = True
            self.btn_entrenar.config(state='disabled')
            
            # Reiniciar barra de progreso
            self.progress_bar['value'] = 0
            self.progress_label.config(text="0%")
            
            # Mostrar parámetros de entrenamiento
            print(f"Parámetros de entrenamiento:")
            print(f"- Alfa: {float(config['alfa'])}")
            print(f"- Épocas máximas: {int(config['max_epocas'])}")
            print(f"- Precisión objetivo: {float(config['precision'])}")
            print(f"- Funciones de activación: {config['funciones_activacion'][0]} (oculta), {config['funciones_activacion'][1]} (salida)")
            if 'leaky relu' in config['funciones_activacion']:
                print(f"- Beta Leaky ReLU: {float(config['beta_leaky_relu'])}")
            if config['momentum']:
                print(f"- Momentum habilitado con Beta: {float(config['beta'])}")

            # Crear red neuronal
            self.red = RedBP(config)
            
            # Iniciar entrenamiento en un hilo separado para no bloquear la interfaz
            import threading
            self.thread_entrenamiento = threading.Thread(target=self.ejecutar_entrenamiento)
            self.thread_entrenamiento.daemon = True
            self.thread_entrenamiento.start()
            
            # Iniciar actualización periódica de la interfaz
            self.root.after(100, self.actualizar_progreso_entrenamiento)

        except Exception as e:
            print(f"Error al iniciar entrenamiento: {str(e)}")
            self.entrenamiento_en_progreso = False
            self.btn_entrenar.config(state='normal')
            import traceback
            print(traceback.format_exc())
            
    def ejecutar_entrenamiento(self):
        """Ejecuta el entrenamiento en un hilo separado"""
        try:
            # Entrenar y obtener errores
            print("Iniciando entrenamiento con backpropagation...")
            
            # Modificar la clase RedBP para que acepte una función de callback
            # que actualice el progreso del entrenamiento
            self.errores_entrenamiento = []
            self.epoca_actual = 0
            self.max_epocas = int(self.max_epocas_input.get())
            
            # Entrenar la red
            self.errores_entrenamiento = self.red.entrenar(
                np.array(self.datos_entrenamiento),
                np.array(self.datos_salida)
            )
            
            # Guardar pesos automáticamente
            carpeta = "Pesos_entrenados"
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
            self.pesos_archivo = os.path.join(carpeta, "pesos_actuales.json")
            self.red.guardar_pesos(self.pesos_archivo)
            
            # Marcar finalización del entrenamiento
            self.entrenamiento_completado = True
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Asegurarse de que la interfaz se actualice al finalizar
            self.entrenamiento_en_progreso = False
            
    def actualizar_progreso_entrenamiento(self):
        """Actualiza la interfaz durante el entrenamiento"""
        if not self.entrenamiento_en_progreso:
            # El entrenamiento ha terminado
            if hasattr(self, 'entrenamiento_completado') and self.entrenamiento_completado:
                # Actualizar estado de entrenamiento
                self.status_label.config(text="Estado: Entrenado Exitosamente", foreground="green")
                self.status_indicator.delete("all")
                self.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualizar métricas
                self.epochs_value.config(text=str(len(self.errores_entrenamiento)))
                self.error_value.config(text=f"{float(self.errores_entrenamiento[-1]):.6f}")
                
                # Mostrar gráfica
                self.graficar_errores(self.errores_entrenamiento)
                print(f"Entrenamiento completado exitosamente en {len(self.errores_entrenamiento)} épocas")
                print(f"Pesos guardados automáticamente en: {self.pesos_archivo}")
                
                # Actualizar barra de progreso al 100%
                self.progress_bar['value'] = 100
                self.progress_label.config(text="100%")
                
                # Limpiar flag
                self.entrenamiento_completado = False
            
            # Habilitar botón de entrenamiento
            self.btn_entrenar.config(state='normal')
            return
            
        # Obtener progreso actual
        if hasattr(self.red, 'epoca_actual') and hasattr(self.red, 'max_epocas'):
            epoca_actual = self.red.epoca_actual
            max_epocas = self.red.max_epocas
            
            # Calcular porcentaje de progreso
            if max_epocas > 0:
                progreso = min(100, int((epoca_actual / max_epocas) * 100))
                
                # Actualizar barra de progreso
                self.progress_bar['value'] = progreso
                self.progress_label.config(text=f"{progreso}%")
                
                # Actualizar también según el error objetivo si está disponible
                if hasattr(self.red, 'error_actual') and hasattr(self.red, 'precision_objetivo'):
                    error_actual = self.red.error_actual
                    precision_objetivo = self.red.precision_objetivo
                    
                    # Calcular progreso basado en el error (inverso)
                    if error_actual > 0 and precision_objetivo > 0:
                        # Calcular qué tan cerca estamos del error objetivo
                        progreso_error = min(100, max(0, int((1 - error_actual/self.red.error_inicial) * 100)))
                        # Usar el mayor de los dos progresos para la barra
                        progreso = max(progreso, progreso_error)
                        self.progress_bar['value'] = progreso
                        self.progress_label.config(text=f"{progreso}% (Error: {error_actual:.6f})")
                
                # Actualizar log cada 100 épocas
                if epoca_actual % 100 == 0 and epoca_actual > 0:
                    print(f"Entrenando... Época {epoca_actual}/{max_epocas} ({progreso}%)")
        
        # Programar la próxima actualización
        self.root.after(100, self.actualizar_progreso_entrenamiento)

    def graficar_errores(self, errores):
        # Limpiar frame anterior
        for widget in self.error_graph_frame.winfo_children():
            widget.destroy()

        # Crear figura
        self.fig_error = plt.figure(figsize=(6, 4))
        plt.plot(errores, color='#004d25', linewidth=2)
        plt.title("Evolución del Error en Backpropagation", fontsize=12)
        plt.xlabel("Épocas", fontsize=10)
        plt.ylabel("Error Promedio", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Integrar gráfica en Tkinter
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=self.error_graph_frame)
        self.canvas_error.draw()
        self.canvas_error.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def guardar_pesos(self):
        if self.red is None:
            print("Error: Primero debe entrenar la red")
            return
            
        archivo = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")])
        
        if archivo:
            try:
                self.red.guardar_pesos(archivo)
                self.pesos_archivo = archivo
                print(f"Pesos guardados en: {archivo}")
            except Exception as e:
                print(f"Error guardando pesos: {str(e)}")

    def cargar_pesos(self):
        """Carga los pesos desde un archivo JSON"""
        archivo = filedialog.askopenfilename(filetypes=[("Archivos JSON", "*.json")])
        if archivo:
            try:
                # Crear una instancia de la red si no existe
                if self.red is None:
                    # Obtener configuración básica
                    config = self.get_config()
                    if config is None:
                        return
                    self.red = RedBP(config)
                
                # Cargar pesos
                self.red.cargar_pesos(archivo)
                self.pesos_archivo = archivo
                self.pesos_cargados = True
                print(f"Pesos cargados exitosamente desde: {archivo}")
                
                # Actualizar interfaz
                nombre_archivo = os.path.basename(archivo)
                print(f"Red lista para clasificar con pesos: {nombre_archivo}")
                
                # Actualizar estado de entrenamiento
                self.status_label.config(text="Estado: Pesos Cargados", foreground="green")
                self.status_indicator.delete("all")
                self.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
            except Exception as e:
                print(f"Error al cargar pesos: {str(e)}")

    def mostrar_patron(self, patron):
        """Muestra un patrón en la cuadrícula 5x7"""
        # Asegurarse de que el patrón tenga el tamaño correcto
        if len(patron) != 35:  # 5x7 = 35
            print(f"Error: El patrón debe tener 35 elementos, pero tiene {len(patron)}")
            return
            
        for i in range(7):  # 7 filas
            for j in range(5):  # 5 columnas
                idx = i * 5 + j  # Índice en el vector unidimensional
                color = COLOR_PRIMARY if patron[idx] == 1 else 'white'
                self.input_grid[i][j].config(bg=color)
