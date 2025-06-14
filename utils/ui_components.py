"""
Componentes de UI personalizados para la aplicación
Universidad de Cundinamarca
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import os
import sys

# Colores refinados de la Universidad de Cundinamarca
COLOR_PRIMARY = "#004d25"       # Verde oscuro del escudo
COLOR_PRIMARY_LIGHT = "#006633" # Verde principal más claro para hover
COLOR_SECONDARY = "#ffd700"     # Amarillo/dorado del escudo
COLOR_ACCENT_RED = "#e60000"    # Rojo del mapa en el centro
COLOR_ACCENT_BLUE = "#66ccff"   # Azul claro del mapa en el centro
COLOR_BG = "#FFFFFF"            # Fondo blanco para contraste
COLOR_TEXT = "#333333"          # Texto oscuro para mejor legibilidad
COLOR_TEXT_SECONDARY = "#666666" # Texto secundario
COLOR_LIGHT_BG = "#f5f5f5"      # Fondo claro para secciones
COLOR_BORDER = "#e0e0e0"        # Color para bordes sutiles
COLOR_SUCCESS = "#28a745"       # Verde para éxito

class ModernButton(tk.Button):
    """Botón con estilo moderno"""
    def __init__(self, parent, text, command=None, width=20, height=1, bg=COLOR_PRIMARY, fg="white", hover_bg=COLOR_PRIMARY_LIGHT, font=("Arial", 10)):
        super().__init__(
            parent, 
            text=text, 
            command=command,
            bg=bg, 
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief=tk.FLAT,
            width=width,
            height=height,
            font=font
        )
        
        # Efecto hover
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        """Efecto al pasar el mouse por encima"""
        if self["state"] != "disabled":
            self.config(bg=COLOR_PRIMARY_LIGHT)
        
    def on_leave(self, e):
        """Efecto al quitar el mouse"""
        if self["state"] != "disabled":
            self.config(bg=COLOR_PRIMARY)

class ToolTip:
    """Tooltip personalizado para widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Crear ventana de tooltip con estilo mejorado
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = tk.Frame(self.tooltip, background=COLOR_PRIMARY, bd=0)
        frame.pack(fill="both", expand=True)
        
        label = tk.Label(frame, text=self.text, justify='left',
                       background=COLOR_PRIMARY, foreground="white",
                       relief="flat", borderwidth=0,
                       font=("Arial", 9), padx=8, pady=4)
        label.pack(ipadx=1)
        
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def setup_styles():
    """Configura los estilos para los widgets ttk"""
    style = ttk.Style()
    
    # Estilo para pestañas
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=COLOR_LIGHT_BG, padding=[10, 2], font=("Arial", 10))
    style.map("TNotebook.Tab", background=[("selected", COLOR_PRIMARY)], foreground=[("selected", "white")])
    
    # Estilo para marcos con etiqueta
    style.configure("TLabelframe", background=COLOR_BG)
    style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_PRIMARY, font=("Arial", 11, "bold"))
    
    # Estilo para combobox
    style.configure("TCombobox", background=COLOR_LIGHT_BG, fieldbackground="white")
    
    # Estilo para barras de progreso
    style.configure("TProgressbar", background=COLOR_PRIMARY, troughcolor=COLOR_LIGHT_BG, borderwidth=0)

def create_graph_figure(figsize=(6, 4), dpi=100):
    """Crea una figura de matplotlib para gráficos"""
    fig = plt.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(COLOR_BG)
    return fig

def embed_matplotlib_plot(parent, figure):
    """Integra un gráfico de matplotlib en un widget de tkinter"""
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas_widget = canvas.get_tk_widget()
    canvas.draw()
    return canvas_widget

def create_info_frame(parent, title):
    """Crea un marco con título para información"""
    # Marco principal
    frame = ttk.LabelFrame(parent, text=title)
    
    # Contenido
    content = tk.Frame(frame, bg=COLOR_LIGHT_BG, padx=10, pady=10)
    content.pack(fill='x')
    
    return frame, content

def create_labeled_entry(parent, label_text, default_value=""):
    """Crea un campo de entrada con etiqueta"""
    container = tk.Frame(parent, bg=COLOR_LIGHT_BG)
    
    label = tk.Label(container, text=label_text, bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
    label.pack(side=tk.LEFT, padx=(0, 5))
    
    entry = tk.Entry(container, width=10)
    entry.insert(0, default_value)
    entry.pack(side=tk.LEFT)
    
    return container, entry

def create_labeled_combobox(parent, label_text, values, default_index=0):
    """Crea un combobox con etiqueta"""
    container = tk.Frame(parent, bg=COLOR_LIGHT_BG)
    
    label = tk.Label(container, text=label_text, bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
    label.pack(side=tk.LEFT, padx=(0, 5))
    
    combobox = ttk.Combobox(container, values=values, state="readonly", width=15)
    combobox.current(default_index)
    combobox.pack(side=tk.LEFT)
    
    return container, combobox

def create_progress_bar(parent):
    """Crea una barra de progreso con etiqueta"""
    container = tk.Frame(parent, bg=COLOR_LIGHT_BG)
    
    label = tk.Label(container, text="Progreso:", bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
    label.pack(side=tk.LEFT, padx=5)
    
    progress_bar = ttk.Progressbar(container, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress_bar.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
    
    progress_label = tk.Label(container, text="0%", bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
    progress_label.pack(side=tk.LEFT, padx=5)
    
    return container, progress_bar, progress_label

def create_status_indicator(parent, label_text):
    """Crea un indicador de estado con etiqueta"""
    container = tk.Frame(parent, bg=COLOR_LIGHT_BG)
    
    label = tk.Label(container, text=label_text, bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
    label.pack(side=tk.LEFT, padx=5)
    
    indicator = tk.Canvas(container, width=15, height=15, bg=COLOR_LIGHT_BG, highlightthickness=0)
    indicator.pack(side=tk.LEFT, padx=5)
    indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
    
    status_label = tk.Label(container, text="No iniciado", bg=COLOR_LIGHT_BG, fg=COLOR_TEXT_SECONDARY)
    status_label.pack(side=tk.LEFT, padx=5)
    
    return container, indicator, status_label

def add_authors_info(self):
      # Crear un marco para la información de autores con estilo mejorado
      authors_frame = tk.Frame(self.main_frame, bg=COLOR_LIGHT_BG, padx=10, pady=5)  # Reducir padding
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
          bg=COLOR_LIGHT_BG,
          fg=COLOR_TEXT
      )
      authors_info.pack(side=tk.LEFT, padx=5)  # Reducir padding


def create_header_frame(parent, show_developers=True):
    """Crea el encabezado con el escudo, título y nombres de desarrolladores"""
    # Marco principal del encabezado
    header_frame = tk.Frame(parent, bg=COLOR_BG)
    
    # Marco para el logo y título
    logo_title_frame = tk.Frame(header_frame, bg=COLOR_BG)
    logo_title_frame.pack(fill='x')
    
    # Intentar cargar el escudo de la universidad
    try:
        logo_path = obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((60, 60), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        
        logo_label = tk.Label(logo_title_frame, image=logo_photo, bg=COLOR_BG)
        logo_label.image = logo_photo  # Mantener referencia
        logo_label.pack(side=tk.LEFT, padx=(10, 15))
    except Exception as e:
        print(f"Error al cargar el logo: {e}")
    
    # Título y subtítulo
    title_frame = tk.Frame(logo_title_frame, bg=COLOR_BG)
    title_frame.pack(side=tk.LEFT)
    
    title_label = tk.Label(
        title_frame, 
        text="BACKPROPAGATION", 
        font=("Arial", 18, "bold"), 
        bg=COLOR_BG, 
        fg=COLOR_PRIMARY
    )
    title_label.pack(anchor='w')
    
    subtitle_label = tk.Label(
        title_frame, 
        text="Universidad de Cundinamarca", 
        font=("Arial", 12), 
        bg=COLOR_BG, 
        fg=COLOR_TEXT
    )
    subtitle_label.pack(anchor='w')
    
    # Línea separadora
    separator = ttk.Separator(header_frame, orient='horizontal')
    separator.pack(fill='x', pady=(5, 0))
    
    # Información de desarrolladores
    if show_developers:
        developers_frame = tk.Frame(header_frame, bg=COLOR_LIGHT_BG, padx=10, pady=5)  # Reducir padding
        developers_frame.pack(side=tk.BOTTOM, fill=tk.X)

        top_border = tk.Frame(developers_frame, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        bottom_border = tk.Frame(developers_frame, bg=COLOR_BORDER, height=1)
        bottom_border.pack(side=tk.BOTTOM, fill='x')

        developers_label = tk.Label(
            developers_frame, 
            text="Desarrollado por: Sergio Leonardo Moscoso Ramirez - Miguel Ángel Pardo Lopez", 
            font=("Arial", 10), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT
        )
        developers_label.pack(side=tk.LEFT, padx=5)  # Reducir padding
    
    return header_frame

def obtener_ruta_relativa(ruta_archivo):
    """Obtiene la ruta relativa de un archivo"""
    if getattr(sys, 'frozen', False):  # Si el programa está empaquetado con PyInstaller
        base_path = sys._MEIPASS       # Carpeta temporal donde PyInstaller extrae los archivos
    else:
        base_path = os.path.abspath(".")  # Carpeta normal en modo desarrollo

    return os.path.join(base_path, ruta_archivo)
