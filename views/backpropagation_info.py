"""
Backpropagation Information View
Universidad de Cundinamarca - Sistema de Información Educativa
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os
import sys
import math
from utils.ui_components import (
    COLOR_BG, COLOR_PRIMARY, COLOR_LIGHT_BG, COLOR_TEXT, 
    COLOR_TEXT_SECONDARY, COLOR_BORDER, COLOR_SECONDARY,
    ModernButton, ToolTip, setup_styles, create_header_frame,
    embed_matplotlib_plot, create_graph_figure
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from matplotlib.ticker import MaxNLocator

class BackpropagationInfoView:
    def __init__(self, root):
        """Initialize the information view with professional styling"""
        self.root = root
        self.root.configure(bg=COLOR_BG)
        
        # Set up custom styles for widgets
        self.setup_custom_styles()
        
        # Create main container with padding
        self.container = tk.Frame(self.root, bg=COLOR_BG)
        self.container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create the content
        self.create_header()
        self.create_navigation()
        self.create_content_area()
        self.create_footer()
        
        # Default to first tab
        self.show_content("intro")
    
    def setup_custom_styles(self):
        """Set up custom styles for a professional look"""
        # Configure ttk styles
        style = ttk.Style()
        
        # Notebook (tabs) styling
        style.configure(
            "Professional.TNotebook", 
            background=COLOR_BG,
            borderwidth=0
        )
        style.configure(
            "Professional.TNotebook.Tab", 
            background=COLOR_LIGHT_BG,
            foreground=COLOR_TEXT,
            padding=[15, 8],
            font=("Segoe UI", 10)
        )
        style.map(
            "Professional.TNotebook.Tab",
            background=[("selected", COLOR_PRIMARY)],
            foreground=[("selected", "white")]
        )
        
        # Frame styling
        style.configure(
            "Professional.TFrame", 
            background=COLOR_BG
        )
        
        # Scrollbar styling
        style.configure(
            "Professional.Vertical.TScrollbar",
            background=COLOR_LIGHT_BG,
            arrowcolor=COLOR_PRIMARY,
            bordercolor=COLOR_BORDER,
            troughcolor=COLOR_LIGHT_BG
        )
    
    def create_header(self):
        """Create an elegant header with university branding"""
        # Header container
        self.header = tk.Frame(self.container, bg=COLOR_BG, height=100)
        self.header.pack(fill='x', pady=(0, 15))
        
        # Left side - Logo and title
        logo_title_frame = tk.Frame(self.header, bg=COLOR_BG)
        logo_title_frame.pack(side="left", fill='y')
        
        # Try to load and enhance university logo
        try:
            logo_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
            original_image = Image.open(logo_path)
            
            # Resize with high quality
            logo_image = original_image.resize((60, 80), Image.LANCZOS)
            
            # Apply subtle enhancements
            enhancer = ImageEnhance.Contrast(logo_image)
            logo_image = enhancer.enhance(1.2)
            
            # Convert to PhotoImage
            logo_photo = ImageTk.PhotoImage(logo_image)
            
            # Display logo
            logo_label = tk.Label(logo_title_frame, image=logo_photo, bg=COLOR_BG)
            logo_label.image = logo_photo  # Keep reference
            logo_label.pack(side="left", padx=(0, 15))
        except Exception as e:
            print(f"Error loading logo: {e}")
        
        # Title and subtitle with refined typography
        title_frame = tk.Frame(logo_title_frame, bg=COLOR_BG)
        title_frame.pack(side="left", pady=10)
        
        # Main title with larger, bolder font
        title_label = tk.Label(
            title_frame, 
            text="REDES NEURONALES ARTIFICIALES", 
            font=("Segoe UI", 18, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title_label.pack(anchor='w')
        
        # Subtitle with lighter weight
        subtitle_label = tk.Label(
            title_frame, 
            text="Universidad de Cundinamarca", 
            font=("Segoe UI", 12), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle_label.pack(anchor='w')
        
        # Right side - Module info
        module_frame = tk.Frame(self.header, bg=COLOR_BG)
        module_frame.pack(side="right", fill='y', padx=15)
        
        module_title = tk.Label(
            module_frame,
            text="Módulo de Aprendizaje",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_BG,
            fg=COLOR_TEXT
        )
        module_title.pack(anchor='e')
        
        module_name = tk.Label(
            module_frame,
            text="Algoritmo de Retropropagación",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_BG,
            fg=COLOR_PRIMARY
        )
        module_name.pack(anchor='e')
        
        # Elegant separator
        separator = tk.Frame(self.container, height=2, bg=COLOR_PRIMARY)
        separator.pack(fill='x', pady=(0, 15))
    
    def create_navigation(self):
        """Create professional navigation tabs"""
        # Navigation container
        self.nav_frame = tk.Frame(self.container, bg=COLOR_BG)
        self.nav_frame.pack(fill='x', pady=(0, 15))
        
        # Define navigation items
        nav_items = [
            {"id": "intro", "text": "Introducción", "icon": "📚"},
            {"id": "algorithm", "text": "Algoritmo", "icon": "⚙️"},
            {"id": "architecture", "text": "Arquitectura", "icon": "🏗️"},
            {"id": "training", "text": "Entrenamiento", "icon": "📈"},
            {"id": "applications", "text": "Aplicaciones", "icon": "🔍"}
        ]
        
        # Create tab buttons
        self.nav_buttons = {}
        
        for item in nav_items:
            # Create button frame with bottom border for selection indicator
            btn_frame = tk.Frame(self.nav_frame, bg=COLOR_BG)
            btn_frame.pack(side="left", padx=5)
            
            # Selection indicator (hidden by default)
            indicator = tk.Frame(btn_frame, height=3, bg=COLOR_PRIMARY)
            
            # Tab button
            btn = tk.Button(
                btn_frame,
                text=f"{item['icon']} {item['text']}",
                font=("Segoe UI", 11),
                bg=COLOR_BG,
                fg=COLOR_TEXT,
                bd=0,
                padx=15,
                pady=8,
                activebackground=COLOR_LIGHT_BG,
                activeforeground=COLOR_PRIMARY,
                cursor="hand2",
                command=lambda id=item["id"]: self.show_content(id)
            )
            btn.pack(fill="x")
            indicator.pack(fill="x", side="bottom")
            
            # Store references
            self.nav_buttons[item["id"]] = {
                "button": btn,
                "indicator": indicator
            }
        
        # Add subtle separator below navigation
        nav_separator = tk.Frame(self.container, height=1, bg=COLOR_BORDER)
        nav_separator.pack(fill='x', pady=(0, 5))

        
    
    def create_content_area(self):
        """Create the main content area with professional styling"""
        # Content container
        self.content_container = tk.Frame(self.container, bg=COLOR_BG)
        self.content_container.pack(fill='both', expand=True)
        
        # Create content frames for each section
        self.content_frames = {
            "intro": self.create_intro_content(),
            "algorithm": self.create_algorithm_content(),
            "architecture": self.create_architecture_content(),
            "training": self.create_training_content(),
            "applications": self.create_applications_content()
        }
        
        # Hide all frames initially
        for frame in self.content_frames.values():
            frame.pack_forget()
    
    def show_content(self, content_id):
        """Show the selected content and update navigation"""
        # Hide all content frames
        for frame in self.content_frames.values():
            frame.pack_forget()
        
        # Reset all navigation buttons
        for nav_id, elements in self.nav_buttons.items():
            elements["button"].config(
                bg=COLOR_BG, 
                fg=COLOR_TEXT
            )
            elements["indicator"].config(bg=COLOR_BG)  # Hide indicator
        
        # Show selected content
        self.content_frames[content_id].pack(fill='both', expand=True)
        
        # Update selected navigation button
        self.nav_buttons[content_id]["button"].config(
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        self.nav_buttons[content_id]["indicator"].config(bg=COLOR_PRIMARY)  # Show indicator
    
    def create_intro_content(self):
        """Create introduction content with professional layout"""
        frame = tk.Frame(self.content_container, bg=COLOR_BG)
        
        # Create scrollable content
        canvas = tk.Canvas(frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            frame, 
            orient="vertical", 
            command=canvas.yview,
            style="Professional.Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Enable mousewheel scrolling without clicking
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mousewheel when mouse enters the canvas
        canvas.bind("<Enter>", _bind_mousewheel)
        # Unbind mousewheel when mouse leaves the canvas
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # Section title with professional typography
        title_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=10)
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame, 
            text="Algoritmo de Retropropagación", 
            font=("Segoe UI", 22, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            title_frame, 
            text="Fundamentos y Aplicaciones en Redes Neuronales", 
            font=("Segoe UI", 14), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle.pack(anchor="w")
        
        # Main content in a professional card layout
        content_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        content_card.pack(fill="x", pady=15)
        
        # Introduction text with professional typography
        intro_text = """El algoritmo de retropropagación (backpropagation) es un método fundamental de aprendizaje supervisado para redes neuronales artificiales. Desarrollado en la década de 1970 y popularizado en los años 80, este algoritmo revolucionó el campo del aprendizaje automático al proporcionar un método eficiente para entrenar redes neuronales multicapa."""
        
        intro_label = tk.Label(
            content_card, 
            text=intro_text, 
            font=("Segoe UI", 12), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        intro_label.pack(anchor="w", pady=(0, 20))
        
        # Key concept with highlight
        key_concept_frame = tk.Frame(content_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        key_concept_frame.pack(fill="x", pady=10)
        
        key_title = tk.Label(
            key_concept_frame, 
            text="CONCEPTO CLAVE", 
            font=("Segoe UI", 10, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        key_title.pack(anchor="w")
        
        key_text = """La retropropagación recibe este nombre porque el error se propaga desde la capa de salida hacia atrás, a través de las capas ocultas, hasta llegar a la capa de entrada. Durante este proceso, los pesos de las conexiones se ajustan para reducir el error en futuras predicciones."""
        
        key_label = tk.Label(
            key_concept_frame, 
            text=key_text, 
            font=("Segoe UI", 11, "italic"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        key_label.pack(anchor="w", pady=(5, 0))
        
        # Two-column layout for content and visualization
        columns_frame = tk.Frame(content_card, bg="white")
        columns_frame.pack(fill="x", pady=20)
        
        # Left column - Additional text
        left_column = tk.Frame(columns_frame, bg="white")
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        additional_text = """El algoritmo de retropropagación utiliza el descenso por gradiente para minimizar la función de costo, que mide la diferencia entre la salida de la red y los valores objetivo. Este proceso iterativo permite que la red "aprenda" de sus errores y mejore gradualmente su rendimiento.

Características principales:

• Aprendizaje supervisado: Requiere ejemplos etiquetados para el entrenamiento
• Ajuste de pesos: Modifica los parámetros de la red para minimizar el error
• Propagación bidireccional: Combina propagación hacia adelante y hacia atrás
• Diferenciabilidad: Requiere funciones de activación diferenciables
• Versatilidad: Aplicable a diversos problemas de clasificación y regresión"""
        
        additional_label = tk.Label(
            left_column, 
            text=additional_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=400
        )
        additional_label.pack(anchor="w")
        
        # Right column - Visualization
        right_column = tk.Frame(columns_frame, bg="white")
        right_column.pack(side="right", fill="both", expand=True)
        
        # Try to load image or create visualization
        try:
            # Get the relative path safely
            image_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "backpropagation_diagram.png"))
            
            # Create image frame with subtle border
            image_frame = tk.Frame(right_column, bg="white", bd=1, relief="solid")
            image_frame.pack(pady=5)
            
            # Load and enhance the image
            image = Image.open(image_path)
            image = image.resize((450, 280), Image.LANCZOS)
            
            # Apply subtle enhancements
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            photo = ImageTk.PhotoImage(image)
            
            # Create a label with the image
            image_label = tk.Label(image_frame, image=photo, bg="white")
            image_label.image = photo  # Keep a reference
            image_label.pack(padx=2, pady=2)
            
            # Add professional caption
            caption = tk.Label(
                right_column, 
                text="Figura 1: Diagrama de una red neuronal con retropropagación", 
                font=("Segoe UI", 9, "italic"), 
                bg="white", 
                fg=COLOR_TEXT_SECONDARY
            )
            caption.pack(pady=(5, 0))
            
        except Exception as e:
            print(f"Error loading image: {e}")
            
            # Create a professional visualization using matplotlib
            fig = create_graph_figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Create a professional neural network diagram
            layer_sizes = [4, 5, 5, 3]
            layer_positions = [1, 2.5, 4, 5.5]
            
            # Background styling
            ax.set_facecolor('white')
            
            # Draw nodes with professional styling
            for i, (size, pos) in enumerate(zip(layer_sizes, layer_positions)):
                layer_name = ["Entrada", "Oculta 1", "Oculta 2", "Salida"][i]
                
                # Add layer label
                ax.text(pos, -1.2, f"Capa {layer_name}", fontsize=9, ha='center', 
                       color=COLOR_TEXT_SECONDARY, fontweight='bold')
                
                for j in range(size):
                    y = (j - (size-1)/2) * 0.5
                    
                    # Node color based on layer type
                    if i == 0:  # Input layer
                        node_color = COLOR_PRIMARY
                        alpha = 0.8
                    elif i == len(layer_sizes)-1:  # Output layer
                        node_color = COLOR_PRIMARY
                        alpha = 0.8
                    else:  # Hidden layers
                        node_color = COLOR_SECONDARY
                        alpha = 0.7
                    
                    # Draw node with subtle gradient effect
                    circle = plt.Circle((pos, y), 0.15, color=node_color, alpha=alpha)
                    ax.add_patch(circle)
                    
                    # Add subtle border
                    border = plt.Circle((pos, y), 0.15, fill=False, edgecolor='black', alpha=0.3)
                    ax.add_patch(border)
                    
                    # Add node labels
                    if i == 0:
                        ax.text(pos-0.3, y, f"x{j+1}", fontsize=9, ha='right', va='center')
                    elif i == len(layer_sizes)-1:
                        ax.text(pos+0.3, y, f"y{j+1}", fontsize=9, ha='left', va='center')
            
            # Draw connections between layers with professional styling
            for i in range(len(layer_sizes)-1):
                for j in range(layer_sizes[i]):
                    y1 = (j - (layer_sizes[i]-1)/2) * 0.5
                    
                    # Only draw a subset of connections for clarity
                    connection_subset = np.random.choice(
                        range(layer_sizes[i+1]), 
                        size=min(3, layer_sizes[i+1]), 
                        replace=False
                    )
                    
                    for k in connection_subset:
                        y2 = (k - (layer_sizes[i+1]-1)/2) * 0.5
                        ax.plot(
                            [layer_positions[i], layer_positions[i+1]], 
                            [y1, y2], 
                            '-', 
                            color='gray', 
                            alpha=0.4,
                            linewidth=0.8
                        )
            
            # Add backpropagation arrow
            arrow = FancyArrowPatch(
                (5.2, -0.8), (1.8, -0.8),
                arrowstyle='-|>',
                color=COLOR_PRIMARY,
                linewidth=2,
                mutation_scale=15
            )
            ax.add_patch(arrow)
            
            # Add error text
            ax.text(3.5, -0.65, "Retropropagación del Error", fontsize=10, 
                   color=COLOR_PRIMARY, ha='center', fontweight='bold')
            
            # Add forward pass arrow
            forward_arrow = FancyArrowPatch(
                (1.8, -0.95), (5.2, -0.95),
                arrowstyle='-|>',
                color=COLOR_SECONDARY,
                linewidth=2,
                mutation_scale=15
            )
            ax.add_patch(forward_arrow)
            
            # Add forward text
            ax.text(3.5, -1.1, "Propagación Hacia Adelante", fontsize=10, 
                   color=COLOR_SECONDARY, ha='center', fontweight='bold')
            
            ax.set_xlim(0.5, 6)
            ax.set_ylim(-1.4, 1.2)
            ax.axis('off')
            
            # Add title
            ax.set_title("Arquitectura de Red Neuronal con Retropropagación", 
                        fontsize=12, pad=10, fontweight='bold')
            
            # Embed the plot in the frame
            canvas_widget = embed_matplotlib_plot(right_column, fig)
            canvas_widget.pack(pady=5)
            
            # Add professional caption
            caption = tk.Label(
                right_column, 
                text="Figura 1: Diagrama de una red neuronal con retropropagación", 
                font=("Segoe UI", 9, "italic"), 
                bg="white", 
                fg=COLOR_TEXT_SECONDARY
            )
            caption.pack(pady=(5, 0))
        
        # Historical context section
        history_frame = tk.Frame(content_card, bg="white", pady=10)
        history_frame.pack(fill="x", pady=(20, 10))
        
        history_title = tk.Label(
            history_frame, 
            text="CONTEXTO HISTÓRICO", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        history_title.pack(anchor="w")
        
        history_text = """Aunque los conceptos fundamentales de la retropropagación fueron descritos por varios investigadores en los años 60 y 70, fue el trabajo de David Rumelhart, Geoffrey Hinton y Ronald Williams en 1986 el que popularizó el algoritmo. Su artículo "Learning representations by back-propagating errors" demostró la eficacia del método y estableció las bases para el desarrollo moderno de las redes neuronales profundas."""
        
        history_label = tk.Label(
            history_frame, 
            text=history_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        history_label.pack(anchor="w", pady=(5, 0))
        
        # Navigation buttons with professional styling
        nav_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=15)
        nav_frame.pack(fill="x")
        
        next_btn = ModernButton(
            nav_frame, 
            text="Explorar el Algoritmo →", 
            command=lambda: self.show_content("algorithm"),
            width=25,
            height=2,
            font=("Segoe UI", 11)
        )
        next_btn.pack(side="right")
        
        return frame
    
    def create_algorithm_content(self):
        """Create algorithm content with professional layout"""
        frame = tk.Frame(self.content_container, bg=COLOR_BG)
        
        # Create scrollable content
        canvas = tk.Canvas(frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            frame, 
            orient="vertical", 
            command=canvas.yview,
            style="Professional.Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Enable mousewheel scrolling without clicking
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mousewheel when mouse enters the canvas
        canvas.bind("<Enter>", _bind_mousewheel)
        # Unbind mousewheel when mouse leaves the canvas
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # Section title with professional typography
        title_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=10)
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame, 
            text="Funcionamiento del Algoritmo", 
            font=("Segoe UI", 22, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            title_frame, 
            text="Proceso Matemático y Computacional", 
            font=("Segoe UI", 14), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle.pack(anchor="w")
        
        # Main phases card
        phases_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        phases_card.pack(fill="x", pady=15)
        
        phases_title = tk.Label(
            phases_card, 
            text="FASES PRINCIPALES", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        phases_title.pack(anchor="w", pady=(0, 15))
        
        # Two-column layout for phases
        phases_columns = tk.Frame(phases_card, bg="white")
        phases_columns.pack(fill="x")
        
        # Forward pass column
        forward_frame = tk.Frame(phases_columns, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        forward_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        forward_title = tk.Label(
            forward_frame, 
            text="1. Propagación Hacia Adelante", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        forward_title.pack(anchor="w", pady=(0, 10))
        
        forward_points = [
            "• Los datos de entrada se propagan a través de la red, desde la capa de entrada hasta la capa de salida.",
            "• Cada neurona recibe entradas, las procesa mediante una función de activación y produce una salida.",
            "• Al final de esta fase, se obtiene la salida de la red para los datos de entrada proporcionados."
        ]
        
        for point in forward_points:
            point_label = tk.Label(
                forward_frame, 
                text=point, 
                font=("Segoe UI", 11), 
                bg=COLOR_LIGHT_BG, 
                fg=COLOR_TEXT,
                justify="left",
                wraplength=350
            )
            point_label.pack(anchor="w", pady=5)
        
        # Backward pass column
        backward_frame = tk.Frame(phases_columns, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        backward_frame.pack(side="right", fill="both", expand=True)
        
        backward_title = tk.Label(
            backward_frame, 
            text="2. Retropropagación del Error", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        backward_title.pack(anchor="w", pady=(0, 10))
        
        backward_points = [
            "• Se calcula el error entre la salida obtenida y la salida deseada.",
            "• Este error se propaga hacia atrás a través de la red.",
            "• Los pesos de las conexiones se ajustan proporcionalmente a su contribución al error.",
            "• Se utiliza la regla de la cadena del cálculo diferencial para determinar cómo cada peso afecta al error total."
        ]
        
        for point in backward_points:
            point_label = tk.Label(
                backward_frame, 
                text=point, 
                font=("Segoe UI", 11), 
                bg=COLOR_LIGHT_BG, 
                fg=COLOR_TEXT,
                justify="left",
                wraplength=350
            )
            point_label.pack(anchor="w", pady=5)
        
        # Iteration note
        iteration_frame = tk.Frame(phases_card, bg="white", pady=15)
        iteration_frame.pack(fill="x")
        
        iteration_text = "El proceso se repite para múltiples ejemplos (épocas de entrenamiento) hasta que el error se reduce a un nivel aceptable o se alcanza un número máximo de iteraciones."
        
        iteration_label = tk.Label(
            iteration_frame, 
            text=iteration_text, 
            font=("Segoe UI", 11, "italic"), 
            bg="white", 
            fg=COLOR_PRIMARY,
            wraplength=800
        )
        iteration_label.pack(fill="x", pady=10)
        
        # Mathematical representation card
        math_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        math_card.pack(fill="x", pady=15)
        
        math_title = tk.Label(
            math_card, 
            text="REPRESENTACIÓN MATEMÁTICA", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        math_title.pack(anchor="w", pady=(0, 15))
        
        # Forward pass equations
        forward_eq_frame = tk.Frame(math_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        forward_eq_frame.pack(fill="x", pady=10)
        
        forward_eq_title = tk.Label(
            forward_eq_frame, 
            text="Propagación hacia adelante:", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        forward_eq_title.pack(anchor="w", pady=(0, 10))
        
        forward_eq_text = """Para cada neurona j en la capa l:
        
z_j^(l) = ∑_i w_ji^(l) * a_i^(l-1) + b_j^(l)
a_j^(l) = f(z_j^(l))

Donde:
- z_j^(l) es la entrada ponderada a la neurona j en la capa l
- w_ji^(l) es el peso de la conexión entre la neurona i en la capa l-1 y la neurona j en la capa l
- a_i^(l-1) es la activación de la neurona i en la capa l-1
- b_j^(l) es el sesgo de la neurona j en la capa l
- f es la función de activación
- a_j^(l) es la activación de la neurona j en la capa l"""
        
        forward_eq_label = tk.Label(
            forward_eq_frame, 
            text=forward_eq_text, 
            font=("Consolas", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left"
        )
        forward_eq_label.pack(anchor="w", pady=5)
        
        # Backward pass equations
        backward_eq_frame = tk.Frame(math_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        backward_eq_frame.pack(fill="x", pady=10)
        
        backward_eq_title = tk.Label(
            backward_eq_frame, 
            text="Retropropagación del error:", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        backward_eq_title.pack(anchor="w", pady=(0, 10))
        
        backward_eq_text = """Para la capa de salida L:
δ_j^(L) = ∂C/∂a_j^(L) * f'(z_j^(L))

Para las capas ocultas l:
δ_j^(l) = (∑_k w_kj^(l+1) * δ_k^(l+1)) * f'(z_j^(l))

Actualización de pesos y sesgos:
∂C/∂w_ji^(l) = a_i^(l-1) * δ_j^(l)
∂C/∂b_j^(l) = δ_j^(l)

w_ji^(l) = w_ji^(l) - α * ∂C/∂w_ji^(l)
b_j^(l) = b_j^(l) - α * ∂C/∂b_j^(l)

Donde:
- δ_j^(l) es el error de la neurona j en la capa l
- C es la función de costo
- f' es la derivada de la función de activación
- α es la tasa de aprendizaje"""
        
        backward_eq_label = tk.Label(
            backward_eq_frame, 
            text=backward_eq_text, 
            font=("Consolas", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left"
        )
        backward_eq_label.pack(anchor="w", pady=5)
        
        # Algorithm visualization
        viz_frame = tk.Frame(math_card, bg="white", pady=15)
        viz_frame.pack(fill="x")
        
        viz_title = tk.Label(
            viz_frame, 
            text="VISUALIZACIÓN DEL ALGORITMO", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        viz_title.pack(anchor="w", pady=(0, 15))
        
        # Create visualization using matplotlib
        fig = create_graph_figure(figsize=(8, 4))
        
        # Error surface plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a more professional error surface
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 * (X**2 + 0.5 * Y**2 + 0.5 * np.sin(X*Y))
        
        # Plot surface with professional coloring
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines on bottom for clarity
        cset = ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
        
        # Add a gradient descent path
        path_x = np.array([1.8, 1.5, 1.0, 0.5, 0.2, 0.0])
        path_y = np.array([1.5, 1.2, 0.8, 0.4, 0.1, 0.0])
        path_z = 0.5 * (path_x**2 + 0.5 * path_y**2 + 0.5 * np.sin(path_x*path_y)) + 0.05
        
        ax.plot(path_x, path_y, path_z, 'ro-', linewidth=2, markersize=5, 
               label='Descenso por Gradiente')
        
        # Add annotations
        ax.text(1.8, 1.5, path_z[0] + 0.2, "Inicio", color='red', fontsize=9)
        ax.text(0.0, 0.0, path_z[-1] + 0.2, "Mínimo", color='red', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel('Peso 1 (w₁)')
        ax.set_ylabel('Peso 2 (w₂)')
        ax.set_zlabel('Error (E)')
        ax.set_title('Superficie de Error y Descenso por Gradiente', pad=10)
        
        # Adjust view angle for better visualization
        ax.view_init(elev=35, azim=-45)
        
        # Embed the plot in the frame
        canvas_widget = embed_matplotlib_plot(viz_frame, fig)
        canvas_widget.pack(pady=10)
        
        # Add professional caption
        caption = tk.Label(
            viz_frame, 
            text="Figura 2: Visualización del descenso por gradiente en una superficie de error", 
            font=("Segoe UI", 9, "italic"), 
            bg="white", 
            fg=COLOR_TEXT_SECONDARY
        )
        caption.pack(pady=(5, 0))
        
        # Pseudocode section
        pseudocode_frame = tk.Frame(scrollable_frame, bg="white", padx=30, pady=30, bd=1, relief="solid")
        pseudocode_frame.pack(fill="x", pady=15)
        
        pseudocode_title = tk.Label(
            pseudocode_frame, 
            text="PSEUDOCÓDIGO DEL ALGORITMO", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        pseudocode_title.pack(anchor="w", pady=(0, 15))
        
        pseudocode_text = """Inicializar pesos y sesgos con valores aleatorios pequeños
Mientras no se cumpla criterio de parada:
    Para cada ejemplo de entrenamiento (x, y):
        # Propagación hacia adelante
        a = x  # Activación de la capa de entrada
        Para cada capa l:
            z[l] = w[l] * a[l-1] + b[l]
            a[l] = f(z[l])  # Aplicar función de activación
        
        # Calcular error
        E = función_de_costo(a[L], y)  # L es la capa de salida
        
        # Retropropagación
        delta[L] = derivada_costo(a[L], y) * derivada_f(z[L])
        Para cada capa l desde L-1 hasta 1:
            delta[l] = (w[l+1]^T * delta[l+1]) * derivada_f(z[l])
        
        # Actualizar pesos y sesgos
        Para cada capa l:
            w[l] = w[l] - tasa_aprendizaje * delta[l] * a[l-1]^T
            b[l] = b[l] - tasa_aprendizaje * delta[l]"""
        
        pseudocode_label = tk.Label(
            pseudocode_frame, 
            text=pseudocode_text, 
            font=("Consolas", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left"
        )
        pseudocode_label.pack(anchor="w", pady=5)
        
        # Navigation buttons with professional styling
        nav_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=15)
        nav_frame.pack(fill="x")
        
        prev_btn = ModernButton(
            nav_frame, 
            text="← Introducción", 
            command=lambda: self.show_content("intro"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        prev_btn.pack(side="left")
        
        next_btn = ModernButton(
            nav_frame, 
            text="Arquitectura →", 
            command=lambda: self.show_content("architecture"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        next_btn.pack(side="right")
        
        return frame
    
    def create_architecture_content(self):
        """Create architecture content with professional layout"""
        frame = tk.Frame(self.content_container, bg=COLOR_BG)
        
        # Create scrollable content
        canvas = tk.Canvas(frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            frame, 
            orient="vertical", 
            command=canvas.yview,
            style="Professional.Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Enable mousewheel scrolling without clicking
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mousewheel when mouse enters the canvas
        canvas.bind("<Enter>", _bind_mousewheel)
        # Unbind mousewheel when mouse leaves the canvas
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # Section title with professional typography
        title_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=10)
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame, 
            text="Arquitectura de la Red Neuronal", 
            font=("Segoe UI", 22, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            title_frame, 
            text="Estructura y Componentes", 
            font=("Segoe UI", 14), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle.pack(anchor="w")
        
        # Main content card
        content_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        content_card.pack(fill="x", pady=15)
        
        # Two-column layout
        columns_frame = tk.Frame(content_card, bg="white")
        columns_frame.pack(fill="x")
        
        # Left column - Text description
        left_column = tk.Frame(columns_frame, bg="white")
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        # Network components section
        components_title = tk.Label(
            left_column, 
            text="COMPONENTES DE LA RED", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        components_title.pack(anchor="w", pady=(0, 15))
        
        components_text = """Una red neuronal típica para el algoritmo de retropropagación consta de:

• Capa de entrada: Recibe los datos de entrada y los transmite a la siguiente capa. El número de neuronas corresponde a la dimensionalidad de los datos de entrada.

• Capas ocultas: Procesan la información. Puede haber una o más capas ocultas. El número y tamaño de estas capas determina la capacidad de la red para aprender representaciones complejas.

• Capa de salida: Produce el resultado final de la red. El número de neuronas depende del tipo de problema (regresión, clasificación binaria, clasificación multiclase).

Cada neurona en la red está conectada a todas las neuronas de la capa siguiente (conexiones totales). Estas conexiones tienen pesos asociados que se ajustan durante el entrenamiento."""
        
        components_label = tk.Label(
            left_column, 
            text=components_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        components_label.pack(anchor="w", pady=5)
        
        # Right column - Interactive visualization
        right_column = tk.Frame(columns_frame, bg="white")
        right_column.pack(side="right", fill="both", expand=True)
        
        # Create visualization using matplotlib
        fig = create_graph_figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        
        # Create a professional neural network diagram
        layer_sizes = [4, 6, 5, 3]
        layer_positions = [1, 2.5, 4, 5.5]
        layer_names = ["Entrada", "Oculta 1", "Oculta 2", "Salida"]
        
        # Background styling
        ax.set_facecolor('white')
        
        # Draw nodes with professional styling
        for i, (size, pos, name) in enumerate(zip(layer_sizes, layer_positions, layer_names)):
            # Add layer label
            ax.text(pos, -1.2, f"Capa {name}", fontsize=10, ha='center', 
                   color=COLOR_TEXT, fontweight='bold')
            
            for j in range(size):
                y = (j - (size-1)/2) * 0.4
                
                # Node color based on layer type
                if i == 0:  # Input layer
                    node_color = COLOR_PRIMARY
                    alpha = 0.8
                elif i == len(layer_sizes)-1:  # Output layer
                    node_color = COLOR_PRIMARY
                    alpha = 0.8
                else:  # Hidden layers
                    node_color = COLOR_SECONDARY
                    alpha = 0.7
                
                # Draw node with subtle gradient effect
                circle = plt.Circle((pos, y), 0.15, color=node_color, alpha=alpha)
                ax.add_patch(circle)
                
                # Add subtle border
                border = plt.Circle((pos, y), 0.15, fill=False, edgecolor='black', alpha=0.3)
                ax.add_patch(border)
                
                # Add node labels
                if i == 0:
                    ax.text(pos-0.3, y, f"x{j+1}", fontsize=9, ha='right', va='center')
                elif i == len(layer_sizes)-1:
                    ax.text(pos+0.3, y, f"y{j+1}", fontsize=9, ha='left', va='center')
        
        # Draw connections between layers with professional styling
        for i in range(len(layer_sizes)-1):
            for j in range(layer_sizes[i]):
                y1 = (j - (layer_sizes[i]-1)/2) * 0.4
                
                # Only draw a subset of connections for clarity
                connection_subset = np.random.choice(
                    range(layer_sizes[i+1]), 
                    size=min(3, layer_sizes[i+1]), 
                    replace=False
                )
                
                for k in connection_subset:
                    y2 = (k - (layer_sizes[i+1]-1)/2) * 0.4
                    ax.plot(
                        [layer_positions[i], layer_positions[i+1]], 
                        [y1, y2], 
                        '-', 
                        color='gray', 
                        alpha=0.4,
                        linewidth=0.8
                    )
        
        # Highlight one path to show weight
        highlight_path_x = [layer_positions[0], layer_positions[1]]
        highlight_path_y = [(0 - (layer_sizes[0]-1)/2) * 0.4, (2 - (layer_sizes[1]-1)/2) * 0.4]
        ax.plot(highlight_path_x, highlight_path_y, '-', color=COLOR_PRIMARY, linewidth=2)
        ax.text(1.75, -0.1, "w_ji", fontsize=10, color=COLOR_PRIMARY, fontweight='bold')
        
        ax.set_xlim(0.5, 6)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Arquitectura de una Red Neuronal Multicapa", fontsize=12, pad=10, fontweight='bold')
        ax.axis('off')
        
        # Embed the plot in the frame
        canvas_widget = embed_matplotlib_plot(right_column, fig)
        canvas_widget.pack(pady=5)
        
        # Add professional caption
        caption = tk.Label(
            right_column, 
            text="Figura 3: Arquitectura típica de una red neuronal multicapa", 
            font=("Segoe UI", 9, "italic"), 
            bg="white", 
            fg=COLOR_TEXT_SECONDARY
        )
        caption.pack(pady=(5, 0))
        
        # Activation functions section
        activation_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        activation_card.pack(fill="x", pady=15)
        
        activation_title = tk.Label(
            activation_card, 
            text="FUNCIONES DE ACTIVACIÓN", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        activation_title.pack(anchor="w", pady=(0, 15))
        
        activation_text = """Cada neurona en la red aplica una función de activación a la suma ponderada de sus entradas. Estas funciones introducen no-linealidades que permiten a la red aprender patrones complejos. Las funciones de activación comunes incluyen:"""
        
        activation_intro = tk.Label(
            activation_card, 
            text=activation_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        activation_intro.pack(anchor="w", pady=(0, 15))
        
        # Create visualization of activation functions
        fig = create_graph_figure(figsize=(8, 4))
        
        # Generate x values
        x = np.linspace(-5, 5, 100)
        
        # Plot multiple activation functions
        ax1 = fig.add_subplot(121)
        
        # Sigmoid
        sigmoid = 1 / (1 + np.exp(-x))
        ax1.plot(x, sigmoid, label='Sigmoide', color=COLOR_PRIMARY, linewidth=2)
        
        # Tanh
        tanh = np.tanh(x)
        ax1.plot(x, tanh, label='Tanh', color=COLOR_SECONDARY, linewidth=2)
        
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Funciones Sigmoidales', fontsize=12, pad=10)
        
        # ReLU variants
        ax2 = fig.add_subplot(122)
        
        # ReLU
        relu = np.maximum(0, x)
        ax2.plot(x, relu, label='ReLU', color='#e60000', linewidth=2)
        
        # Leaky ReLU
        leaky_relu = np.maximum(0.1 * x, x)
        ax2.plot(x, leaky_relu, label='Leaky ReLU', color='#66ccff', linewidth=2)
        
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('Funciones ReLU', fontsize=12, pad=10)
        
        # Adjust layout
        fig.tight_layout(pad=3.0)
        
        # Embed the plot in the frame
        canvas_widget = embed_matplotlib_plot(activation_card, fig)
        canvas_widget.pack(pady=15)
        
        # Add professional caption
        caption = tk.Label(
            activation_card, 
            text="Figura 4: Funciones de activación comunes en redes neuronales", 
            font=("Segoe UI", 9, "italic"), 
            bg="white", 
            fg=COLOR_TEXT_SECONDARY
        )
        caption.pack(pady=(0, 15))
        
        # Activation functions descriptions
        functions_frame = tk.Frame(activation_card, bg="white")
        functions_frame.pack(fill="x")
        
        # Left column
        left_functions = tk.Frame(functions_frame, bg="white")
        left_functions.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        sigmoid_title = tk.Label(
            left_functions, 
            text="Sigmoide: f(x) = 1/(1+e^(-x))", 
            font=("Segoe UI", 11, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        sigmoid_title.pack(anchor="w", pady=(0, 5))
        
        sigmoid_desc = tk.Label(
            left_functions, 
            text="Mapea valores a un rango entre 0 y 1. Útil para problemas de clasificación binaria y en la capa de salida para probabilidades.", 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        sigmoid_desc.pack(anchor="w", pady=(0, 10))
        
        tanh_title = tk.Label(
            left_functions, 
            text="Tangente hiperbólica: f(x) = tanh(x)", 
            font=("Segoe UI", 11, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        tanh_title.pack(anchor="w", pady=(0, 5))
        
        tanh_desc = tk.Label(
            left_functions, 
            text="Mapea valores a un rango entre -1 y 1. Similar a la sigmoide pero con salidas centradas en cero, lo que facilita el aprendizaje en capas profundas.", 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        tanh_desc.pack(anchor="w")
        
        # Right column
        right_functions = tk.Frame(functions_frame, bg="white")
        right_functions.pack(side="right", fill="both", expand=True)
        
        relu_title = tk.Label(
            right_functions, 
            text="ReLU: f(x) = max(0, x)", 
            font=("Segoe UI", 11, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        relu_title.pack(anchor="w", pady=(0, 5))
        
        relu_desc = tk.Label(
            right_functions, 
            text="Función lineal rectificada. Computacionalmente eficiente y ayuda a mitigar el problema del desvanecimiento del gradiente. Es la función más utilizada en redes profundas modernas.", 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        relu_desc.pack(anchor="w", pady=(0, 10))
        
        leaky_title = tk.Label(
            right_functions, 
            text="Leaky ReLU: f(x) = max(αx, x)", 
            font=("Segoe UI", 11, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        leaky_title.pack(anchor="w", pady=(0, 5))
        
        leaky_desc = tk.Label(
            right_functions, 
            text="Variante de ReLU que permite un pequeño gradiente cuando la unidad no está activa (x < 0), ayudando a prevenir el problema de 'neuronas muertas'.", 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        leaky_desc.pack(anchor="w")
        
        # Navigation buttons with professional styling
        nav_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=15)
        nav_frame.pack(fill="x")
        
        prev_btn = ModernButton(
            nav_frame, 
            text="← Algoritmo", 
            command=lambda: self.show_content("algorithm"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        prev_btn.pack(side="left")
        
        next_btn = ModernButton(
            nav_frame, 
            text="Entrenamiento →", 
            command=lambda: self.show_content("training"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        next_btn.pack(side="right")
        
        return frame
    
    def create_training_content(self):
        """Create training process content with professional layout"""
        frame = tk.Frame(self.content_container, bg=COLOR_BG)
        
        # Create scrollable content
        canvas = tk.Canvas(frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            frame, 
            orient="vertical", 
            command=canvas.yview,
            style="Professional.Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Enable mousewheel scrolling without clicking
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mousewheel when mouse enters the canvas
        canvas.bind("<Enter>", _bind_mousewheel)
        # Unbind mousewheel when mouse leaves the canvas
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # Section title with professional typography
        title_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=10)
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame, 
            text="Proceso de Entrenamiento", 
            font=("Segoe UI", 22, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            title_frame, 
            text="Optimización y Ajuste de Parámetros", 
            font=("Segoe UI", 14), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle.pack(anchor="w")
        
        # Training steps card
        steps_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        steps_card.pack(fill="x", pady=15)
        
        steps_title = tk.Label(
            steps_card, 
            text="PASOS DEL ENTRENAMIENTO", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        steps_title.pack(anchor="w", pady=(0, 15))
        
        # Step 1: Initialization
        step1_frame = tk.Frame(steps_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        step1_frame.pack(fill="x", pady=10)
        
        step1_title = tk.Label(
            step1_frame, 
            text="1. Inicialización", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        step1_title.pack(anchor="w", pady=(0, 10))
        
        step1_text = """Los pesos de las conexiones se inicializan con valores aleatorios pequeños. Una buena inicialización es crucial para el entrenamiento efectivo:

• Inicialización Xavier/Glorot: Diseñada para mantener la varianza de las activaciones y gradientes a través de las capas.
• Inicialización He: Variante optimizada para funciones de activación ReLU.

Los sesgos (bias) generalmente se inicializan a cero o a valores pequeños positivos."""
        
        step1_label = tk.Label(
            step1_frame, 
            text=step1_text, 
            font=("Segoe UI", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        step1_label.pack(anchor="w", pady=5)
        
        # Step 2: Forward propagation
        step2_frame = tk.Frame(steps_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        step2_frame.pack(fill="x", pady=10)
        
        step2_title = tk.Label(
            step2_frame, 
            text="2. Propagación hacia adelante", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        step2_title.pack(anchor="w", pady=(0, 10))
        
        step2_text = """• Se presenta un patrón de entrada a la red.
• Se calcula la salida de cada neurona en cada capa aplicando la suma ponderada de entradas y la función de activación.
• Se obtiene la salida final de la red.

Este proceso transforma los datos de entrada a través de múltiples transformaciones no lineales hasta producir una predicción."""
        
        step2_label = tk.Label(
            step2_frame, 
            text=step2_text, 
            font=("Segoe UI", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        step2_label.pack(anchor="w", pady=5)
        
        # Step 3: Error calculation
        step3_frame = tk.Frame(steps_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        step3_frame.pack(fill="x", pady=10)
        
        step3_title = tk.Label(
            step3_frame, 
            text="3. Cálculo del error", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        step3_title.pack(anchor="w", pady=(0, 10))
        
        step3_text = """• Se compara la salida obtenida con la salida deseada.
• Se calcula el error utilizando una función de costo apropiada:
  - Error cuadrático medio (MSE): Para problemas de regresión
  - Entropía cruzada: Para problemas de clasificación
  - Error absoluto medio (MAE): Para problemas menos sensibles a valores atípicos

La función de costo cuantifica qué tan lejos están las predicciones de la red de los valores objetivo."""
        
        step3_label = tk.Label(
            step3_frame, 
            text=step3_text, 
            font=("Segoe UI", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        step3_label.pack(anchor="w", pady=5)
        
        # Step 4: Backpropagation
        step4_frame = tk.Frame(steps_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        step4_frame.pack(fill="x", pady=10)
        
        step4_title = tk.Label(
            step4_frame, 
            text="4. Retropropagación", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        step4_title.pack(anchor="w", pady=(0, 10))
        
        step4_text = """• El error se propaga desde la capa de salida hacia atrás.
• Se calculan los gradientes del error con respecto a cada peso utilizando la regla de la cadena.
• Se ajustan los pesos utilizando la regla del descenso por gradiente:
  Δw = -α * ∂E/∂w
  donde α es la tasa de aprendizaje y ∂E/∂w es el gradiente del error respecto al peso.

Este paso es el corazón del algoritmo, permitiendo que la red aprenda de sus errores."""
        
        step4_label = tk.Label(
            step4_frame, 
            text=step4_text, 
            font=("Segoe UI", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        step4_label.pack(anchor="w", pady=5)
        
        # Step 5: Repetition
        step5_frame = tk.Frame(steps_card, bg=COLOR_LIGHT_BG, padx=25, pady=20)
        step5_frame.pack(fill="x", pady=10)
        
        step5_title = tk.Label(
            step5_frame, 
            text="5. Repetición", 
            font=("Segoe UI", 12, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        step5_title.pack(anchor="w", pady=(0, 10))
        
        step5_text = """• Los pasos 2-4 se repiten para todos los patrones de entrenamiento (un lote o batch).
• Una vez que todos los patrones han sido procesados, se completa una época.
• El entrenamiento continúa durante múltiples épocas hasta que:
  - El error se reduce a un nivel aceptable
  - El rendimiento en un conjunto de validación deja de mejorar
  - Se alcanza un número máximo de épocas

El proceso iterativo permite que la red refine gradualmente sus pesos para minimizar el error."""
        
        step5_label = tk.Label(
            step5_frame, 
            text=step5_text, 
            font=("Segoe UI", 11), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        step5_label.pack(anchor="w", pady=5)
        
        # Training visualization
        viz_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        viz_card.pack(fill="x", pady=15)
        
        viz_title = tk.Label(
            viz_card, 
            text="VISUALIZACIÓN DEL ENTRENAMIENTO", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        viz_title.pack(anchor="w", pady=(0, 15))
        
        # Create visualization using matplotlib
        fig = create_graph_figure(figsize=(8, 4))
        
        # Error curve
        ax1 = fig.add_subplot(121)
        epochs = np.arange(1, 101)
        
        # Training error
        train_error = 1.0 / (1 + 0.05 * epochs) + 0.1 * np.exp(-0.1 * epochs) * np.sin(0.5 * epochs)
        ax1.plot(epochs, train_error, color=COLOR_PRIMARY, linewidth=2, label='Entrenamiento')
        
        # Validation error
        val_error = train_error + 0.1 + 0.05 * np.random.randn(len(epochs))
        val_error = np.maximum(val_error, train_error)  # Validation error >= training error
        ax1.plot(epochs, val_error, color=COLOR_SECONDARY, linewidth=2, linestyle='--', label='Validación')
        
        # Highlight overfitting region
        overfitting_start = 70
        ax1.axvspan(overfitting_start, 100, alpha=0.2, color='red')
        ax1.text(85, 0.5, "Sobreajuste", color='red', fontsize=9, ha='center')
        
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Error')
        ax1.set_title('Curva de Error Durante el Entrenamiento')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Learning rate effect
        ax2 = fig.add_subplot(122)
        
        # Generate data for different learning rates
        x = np.linspace(0, 10, 100)
        
        # Too small learning rate
        small_lr = 2 - 1.8 * np.exp(-0.1 * x)
        ax2.plot(x, small_lr, label='α muy pequeña', color='blue', linewidth=2)
        
        # Good learning rate
        good_lr = 2 - 1.8 * np.exp(-0.3 * x)
        ax2.plot(x, good_lr, label='α óptima', color=COLOR_PRIMARY, linewidth=2)
        
        # Too large learning rate
        large_lr = 2 - 1.8 * np.exp(-0.5 * x) + 0.2 * np.sin(x)
        ax2.plot(x, large_lr, label='α muy grande', color='red', linewidth=2)
        
        ax2.set_xlabel('Iteraciones')
        ax2.set_ylabel('Error')
        ax2.set_title('Efecto de la Tasa de Aprendizaje')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the plot in the frame
        canvas_widget = embed_matplotlib_plot(viz_card, fig)
        canvas_widget.pack(pady=10)
        
        # Add professional caption
        caption = tk.Label(
            viz_card, 
            text="Figura 5: Evolución del error durante el entrenamiento y efecto de la tasa de aprendizaje", 
            font=("Segoe UI", 9, "italic"), 
            bg="white", 
            fg=COLOR_TEXT_SECONDARY
        )
        caption.pack(pady=(5, 15))
        
        # Additional techniques section
        techniques_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        techniques_card.pack(fill="x", pady=15)
        
        techniques_title = tk.Label(
            techniques_card, 
            text="TÉCNICAS AVANZADAS DE ENTRENAMIENTO", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        techniques_title.pack(anchor="w", pady=(0, 15))
        
        # Two-column layout for techniques
        techniques_frame = tk.Frame(techniques_card, bg="white")
        techniques_frame.pack(fill="x")
        
        # Left column
        left_techniques = tk.Frame(techniques_frame, bg="white")
        left_techniques.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        momentum_title = tk.Label(
            left_techniques, 
            text="Momentum", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        momentum_title.pack(anchor="w", pady=(0, 5))
        
        momentum_text = """Añade una fracción del cambio de peso anterior al cambio actual, ayudando a superar mínimos locales y acelerando la convergencia en regiones planas de la superficie de error.

Δw(t) = -α * ∂E/∂w + β * Δw(t-1)

donde β es el coeficiente de momentum (típicamente 0.9)."""
        
        momentum_label = tk.Label(
            left_techniques, 
            text=momentum_text, 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        momentum_label.pack(anchor="w", pady=(0, 15))
        
        regularization_title = tk.Label(
            left_techniques, 
            text="Regularización", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        regularization_title.pack(anchor="w", pady=(0, 5))
        
        regularization_text = """Técnicas para prevenir el sobreajuste:

• L1 Regularización: Añade |w| a la función de costo, promoviendo la dispersión (muchos pesos cercanos a cero).

• L2 Regularización: Añade w² a la función de costo, penalizando pesos grandes.

• Dropout: Desactiva aleatoriamente neuronas durante el entrenamiento, forzando a la red a aprender representaciones más robustas."""
        
        regularization_label = tk.Label(
            left_techniques, 
            text=regularization_text, 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        regularization_label.pack(anchor="w")
        
        # Right column
        right_techniques = tk.Frame(techniques_frame, bg="white")
        right_techniques.pack(side="right", fill="both", expand=True)
        
        adaptive_title = tk.Label(
            right_techniques, 
            text="Tasa de Aprendizaje Adaptativa", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        adaptive_title.pack(anchor="w", pady=(0, 5))
        
        adaptive_text = """Algoritmos que ajustan la tasa de aprendizaje durante el entrenamiento:

• AdaGrad: Adapta la tasa para cada parámetro basándose en los gradientes históricos.

• RMSProp: Utiliza una media móvil ponderada de gradientes cuadrados para normalizar el gradiente.

• Adam: Combina las ideas de momentum y RMSProp, manteniendo medias móviles del gradiente y su cuadrado."""
        
        adaptive_label = tk.Label(
            right_techniques, 
            text=adaptive_text, 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        adaptive_label.pack(anchor="w", pady=(0, 15))
        
        batch_title = tk.Label(
            right_techniques, 
            text="Entrenamiento por Lotes", 
            font=("Segoe UI", 12, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        batch_title.pack(anchor="w", pady=(0, 5))
        
        batch_text = """Estrategias para procesar los datos de entrenamiento:

• Batch: Actualiza los pesos después de procesar todos los ejemplos.

• Estocástico: Actualiza los pesos después de cada ejemplo individual.

• Mini-batch: Actualiza los pesos después de procesar un subconjunto de ejemplos, equilibrando eficiencia y estocasticidad."""
        
        batch_label = tk.Label(
            right_techniques, 
            text=batch_text, 
            font=("Segoe UI", 10), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=350
        )
        batch_label.pack(anchor="w")
        
        # Navigation buttons with professional styling
        nav_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=15)
        nav_frame.pack(fill="x")
        
        prev_btn = ModernButton(
            nav_frame, 
            text="← Arquitectura", 
            command=lambda: self.show_content("architecture"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        prev_btn.pack(side="left")
        
        next_btn = ModernButton(
            nav_frame, 
            text="Aplicaciones →", 
            command=lambda: self.show_content("applications"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        next_btn.pack(side="right")
        
        return frame
    
    def create_applications_content(self):
        """Create applications content with professional layout"""
        frame = tk.Frame(self.content_container, bg=COLOR_BG)
        
        # Create scrollable content
        canvas = tk.Canvas(frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            frame, 
            orient="vertical", 
            command=canvas.yview,
            style="Professional.Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Enable mousewheel scrolling without clicking
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mousewheel when mouse enters the canvas
        canvas.bind("<Enter>", _bind_mousewheel)
        # Unbind mousewheel when mouse leaves the canvas
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # Section title with professional typography
        title_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=10)
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame, 
            text="Aplicaciones", 
            font=("Segoe UI", 22, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_PRIMARY
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            title_frame, 
            text="Casos de Uso y Ejemplos Prácticos", 
            font=("Segoe UI", 14), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT_SECONDARY
        )
        subtitle.pack(anchor="w")
        
        # Introduction card
        intro_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        intro_card.pack(fill="x", pady=15)
        
        intro_text = """El algoritmo de retropropagación ha sido fundamental en el desarrollo de aplicaciones de inteligencia artificial. A pesar de la aparición de algoritmos más avanzados, sigue siendo la base de muchas técnicas modernas de aprendizaje profundo. A continuación se presentan algunas de las aplicaciones más relevantes en diversos campos."""
        
        intro_label = tk.Label(
            intro_card, 
            text=intro_text, 
            font=("Segoe UI", 12), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        intro_label.pack(fill="x")
        
        # Applications grid
        applications_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        applications_card.pack(fill="x", pady=15)
        
        applications_title = tk.Label(
            applications_card, 
            text="CAMPOS DE APLICACIÓN", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        applications_title.pack(anchor="w", pady=(0, 20))
        
        # Applications grid
        grid_frame = tk.Frame(applications_card, bg="white")
        grid_frame.pack(fill="x")
        
        # Define applications with professional descriptions
        applications = [
            {
                "title": "Visión por Computadora",
                "description": "Reconocimiento de objetos, detección facial, segmentación de imágenes médicas, sistemas de vigilancia inteligente y vehículos autónomos.",
                "icon": "🔍"
            },
            {
                "title": "Procesamiento de Lenguaje Natural",
                "description": "Traducción automática, análisis de sentimientos, generación de texto, chatbots y sistemas de respuesta a preguntas.",
                "icon": "💬"
            },
            {
                "title": "Sistemas de Recomendación",
                "description": "Sugerencias personalizadas en plataformas de comercio electrónico, servicios de streaming y redes sociales basadas en preferencias del usuario.",
                "icon": "👍"
            },
            {
                "title": "Análisis Financiero",
                "description": "Predicción de mercados bursátiles, detección de fraudes, evaluación de riesgos crediticios y optimización de carteras de inversión.",
                "icon": "📈"
            },
            {
                "title": "Medicina y Salud",
                "description": "Diagnóstico asistido por computadora, análisis de imágenes médicas, descubrimiento de fármacos y medicina personalizada.",
                "icon": "🏥"
            },
            {
                "title": "Robótica y Control",
                "description": "Sistemas de navegación autónoma, control de brazos robóticos, drones inteligentes y sistemas de automatización industrial.",
                "icon": "🤖"
            },
            {
                "title": "Procesamiento de Señales",
                "description": "Reconocimiento de voz, identificación de hablantes, análisis de audio y filtrado de ruido en telecomunicaciones.",
                "icon": "🔊"
            },
            {
                "title": "Ciencia e Investigación",
                "description": "Análisis de datos genómicos, predicción de estructuras de proteínas, simulaciones climáticas y descubrimiento de nuevos materiales.",
                "icon": "🔬"
            }
        ]
        
        # Create a professional grid of application cards
        row, col = 0, 0
        for app in applications:
            # Create card frame with subtle shadow effect
            card_frame = tk.Frame(grid_frame, bg="white", bd=1, relief="solid", padx=15, pady=15)
            card_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Add subtle top border accent
            accent = tk.Frame(card_frame, bg=COLOR_PRIMARY, height=3)
            accent.pack(fill="x", side="top")
            
            # Icon and title in the same row
            header_frame = tk.Frame(card_frame, bg="white")
            header_frame.pack(fill="x", pady=(10, 10))
            
            # Icon
            icon_label = tk.Label(
                header_frame, 
                text=app["icon"], 
                font=("Segoe UI", 24), 
                bg="white"
            )
            icon_label.pack(side="left")
            
            # Title
            title_label = tk.Label(
                header_frame, 
                text=app["title"], 
                font=("Segoe UI", 12, "bold"), 
                bg="white", 
                fg=COLOR_PRIMARY
            )
            title_label.pack(side="left", padx=(10, 0))
            
            # Description
            desc_label = tk.Label(
                card_frame, 
                text=app["description"], 
                font=("Segoe UI", 10), 
                bg="white", 
                fg=COLOR_TEXT,
                wraplength=320,
                justify="left"
            )
            desc_label.pack(fill="x")
            
            # Update grid position
            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(4):  # 4 rows
            grid_frame.grid_rowconfigure(i, weight=1)
        for i in range(2):  # 2 columns
            grid_frame.grid_columnconfigure(i, weight=1)
        
        # Case study card
        case_study_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        case_study_card.pack(fill="x", pady=15)
        
        case_study_title = tk.Label(
            case_study_card, 
            text="CASO DE ESTUDIO: RECONOCIMIENTO DE PATRONES", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        case_study_title.pack(anchor="w", pady=(0, 15))
        
        case_study_text = """En este laboratorio, exploraremos la implementación y aplicación del algoritmo de retropropagación para el reconocimiento de patrones y clasificación de imágenes. Específicamente, trabajaremos con un conjunto de datos de imágenes de vocales para desarrollar un sistema de reconocimiento.

El proceso incluirá:

1. Preprocesamiento de imágenes para normalización y extracción de características
2. Diseño de una arquitectura de red neuronal apropiada
3. Entrenamiento del modelo utilizando el algoritmo de retropropagación
4. Evaluación del rendimiento y ajuste de hiperparámetros
5. Prueba del sistema con nuevas imágenes

Este caso práctico ilustra cómo el algoritmo de retropropagación puede aplicarse a problemas reales de reconocimiento de patrones, sentando las bases para aplicaciones más complejas en visión por computadora."""
        
        case_study_label = tk.Label(
            case_study_card, 
            text=case_study_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        case_study_label.pack(anchor="w", pady=5)
        
        # Future trends card
        future_card = tk.Frame(
            scrollable_frame, 
            bg="white", 
            padx=30, 
            pady=30,
            bd=1,
            relief="solid"
        )
        future_card.pack(fill="x", pady=15)
        
        future_title = tk.Label(
            future_card, 
            text="TENDENCIAS FUTURAS", 
            font=("Segoe UI", 14, "bold"), 
            bg="white", 
            fg=COLOR_PRIMARY
        )
        future_title.pack(anchor="w", pady=(0, 15))
        
        future_text = """Aunque el algoritmo de retropropagación básico ha sido mejorado con numerosas variantes, sigue siendo fundamental en el campo del aprendizaje profundo. Las tendencias actuales incluyen:

• Arquitecturas más profundas y complejas con conexiones residuales y atención
• Técnicas de optimización avanzadas como aprendizaje por transferencia
• Implementaciones eficientes para dispositivos con recursos limitados
• Integración con otros paradigmas como el aprendizaje por refuerzo
• Aplicaciones en campos emergentes como la computación cuántica

El estudio del algoritmo de retropropagación proporciona una base sólida para comprender estos avances y contribuir al desarrollo futuro de la inteligencia artificial."""
        
        future_label = tk.Label(
            future_card, 
            text=future_text, 
            font=("Segoe UI", 11), 
            bg="white", 
            fg=COLOR_TEXT,
            justify="left",
            wraplength=800
        )
        future_label.pack(anchor="w", pady=5)
        
        # Navigation buttons with professional styling
        nav_frame = tk.Frame(scrollable_frame, bg=COLOR_BG, pady=15)
        nav_frame.pack(fill="x")
        
        prev_btn = ModernButton(
            nav_frame, 
            text="← Entrenamiento", 
            command=lambda: self.show_content("training"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        prev_btn.pack(side="left")
        
        home_btn = ModernButton(
            nav_frame, 
            text="Inicio", 
            command=lambda: self.show_content("intro"),
            width=15,
            height=2,
            font=("Segoe UI", 11)
        )
        home_btn.pack(side="right")
        
        return frame
    
    def create_footer(self):
        """Create a professional footer with university branding"""
        # Footer container
        self.footer = tk.Frame(self.container, bg=COLOR_PRIMARY, height=40)
        self.footer.pack(side=tk.BOTTOM, fill='x')
        
        # Add subtle top border
        top_border = tk.Frame(self.footer, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        # Footer content with flex layout
        footer_content = tk.Frame(self.footer, bg=COLOR_PRIMARY)
        footer_content.pack(fill='both', expand=True, padx=20)
        
        # Copyright info
        copyright_label = tk.Label(
            footer_content, 
            text="© Universidad de Cundinamarca - Simulador de BackPropagation", 
            font=("Segoe UI", 9), 
            bg=COLOR_PRIMARY, 
            fg="white"
        )
        copyright_label.pack(side="left", pady=10)
        
        # Authors info
        authors_label = tk.Label(
            footer_content, 
            text="Desarrollado por: Sergio L. Moscoso R. - Miguel Á. Pardo L.", 
            font=("Segoe UI", 9), 
            bg=COLOR_PRIMARY, 
            fg="white"
        )
        authors_label.pack(side="right", pady=10)
    
    def obtener_ruta_relativa(self, ruta_archivo):
        """Gets the relative path of a file"""
        if getattr(sys, 'frozen', False):  # If the program is packaged with PyInstaller
            base_path = sys._MEIPASS       # Temporary folder where PyInstaller extracts files
        else:
            base_path = os.path.abspath(".")  # Normal folder in development mode

        return os.path.join(base_path, ruta_archivo)
    
