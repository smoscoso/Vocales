import tkinter as tk
from tkinter import ttk
import os
import sys

# Import the views and controllers
from controllers.home_controller import HomeController
from views.main_view import MainView
from controllers.images_controller import AppController
from utils.ui_components import COLOR_BG, COLOR_LIGHT_BG, COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_SECONDARY, ModernButton

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Backpropagation - Universidad de Cundinamarca")
        self.root.geometry("1200x720")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(1000, 690)
        
        # Create the main layout
        self.create_main_layout()
        
        # Initialize controllers and views (but don't create them yet)
        self.home_controller = None
        self.backprop_controller = None
        self.images_controller = None
        
        # Current active view
        self.active_view = None
        
        # Show the default view (Info view with backpropagation description)
        self.show_info()
        
    def create_main_layout(self):
        """Creates the main layout with sidebar and content area"""
        # Main panel
        self.main_panel = tk.Frame(self.root, bg=COLOR_LIGHT_BG)
        self.main_panel.pack(fill='both', expand=True)
        
        # Left sidebar
        self.sidebar = tk.Frame(self.main_panel, bg=COLOR_LIGHT_BG, width=200)
        self.sidebar.pack(side=tk.LEFT, fill='y', padx=0, pady=0)
        self.sidebar.pack_propagate(False)  # Prevent resizing
        
        # Sidebar title
        title_frame = tk.Frame(self.sidebar, bg=COLOR_LIGHT_BG, height=40)
        title_frame.pack(fill='x', padx=0, pady=0)
        
        title_label = tk.Label(title_frame, text="LABORATORIOS", 
                          font=("Arial", 10, "bold"), 
                          bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        title_label.pack(pady=10)
        
        # Horizontal separator
        separator = ttk.Separator(self.sidebar, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=5)
        
        # Navigation buttons - Estilo actualizado para que se parezca a la imagen
        self.btn_info = tk.Button(
            self.sidebar, 
            text="Inicio", 
            command=self.show_info,
            bg=COLOR_PRIMARY, 
            fg='white',
            activebackground=COLOR_PRIMARY_LIGHT,
            activeforeground='white',
            width=20,
            height=2,
            font=("Arial", 9),
            relief=tk.FLAT,
            bd=0
        )
        self.btn_info.pack(pady=5, padx=10)
        
        self.btn_lab3 = tk.Button(
            self.sidebar, 
            text="Laboratorio #3", 
            command=self.show_lab3,
            bg=COLOR_PRIMARY, 
            fg='white',
            activebackground=COLOR_PRIMARY_LIGHT,
            activeforeground='white',
            width=20,
            height=2,
            font=("Arial", 9),
            relief=tk.FLAT,
            bd=0
        )
        self.btn_lab3.pack(pady=5, padx=10)
        
        self.btn_lab3a = tk.Button(
            self.sidebar, 
            text="Laboratorio #3a", 
            command=self.show_lab3a,
            bg=COLOR_PRIMARY, 
            fg='white',
            activebackground=COLOR_PRIMARY_LIGHT,
            activeforeground='white',
            width=20,
            height=2,
            font=("Arial", 9),
            relief=tk.FLAT,
            bd=0
        )
        self.btn_lab3a.pack(pady=5, padx=10)
        
        # Vertical separator line
        separator_line = tk.Frame(self.main_panel, bg=COLOR_PRIMARY, width=3)
        separator_line.pack(side=tk.LEFT, fill='y', padx=0, pady=0)
        
        # Main content area
        self.content_frame = tk.Frame(self.main_panel, bg=COLOR_BG)
        self.content_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=0, pady=0)

    def show_info(self):
        """Shows the information view with backpropagation description"""
        # Highlight active button
        self.btn_info.config(bg=COLOR_SECONDARY, fg=COLOR_PRIMARY)
        self.btn_lab3.config(bg=COLOR_PRIMARY, fg='white')
        self.btn_lab3a.config(bg=COLOR_PRIMARY, fg='white')
        
        # Clear content area
        self.clear_content()
        
        # Create or show the info view
        if self.home_controller is None:
            # Create a new frame for this view
            view_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
            view_frame.pack(fill='both', expand=True)
            
            # Initialize the controller with this frame
            self.home_controller = HomeController(view_frame)
            
            # Store reference to the view's main frame
            self.home_view_frame = view_frame
        else:
            # Show the existing view
            self.home_view_frame.pack(fill='both', expand=True)
        
        self.active_view = "info"

    def show_lab3(self):
        """Shows the Backpropagation Lab view (main_view.py)"""
        # Highlight active button
        self.btn_info.config(bg=COLOR_PRIMARY, fg='white')
        self.btn_lab3.config(bg=COLOR_SECONDARY, fg=COLOR_PRIMARY)
        self.btn_lab3a.config(bg=COLOR_PRIMARY, fg='white')
        
        # Clear content area
        self.clear_content()
        
        # Create or show the backpropagation view
        if self.backprop_controller is None:
            # Create a new frame for this view
            view_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
            view_frame.pack(fill='both', expand=True)
            
            # Initialize the controller with this frame
            self.backprop_controller = MainView(view_frame)
            
            # Store reference to the view's main frame
            self.backprop_view_frame = view_frame
        else:
            # Show the existing view
            self.backprop_view_frame.pack(fill='both', expand=True)
        
        self.active_view = "lab3"

    def show_lab3a(self):
        """Shows the Images Lab view (main_view_Images.py)"""
        # Highlight active button
        self.btn_info.config(bg=COLOR_PRIMARY, fg='white')
        self.btn_lab3.config(bg=COLOR_PRIMARY, fg='white')
        self.btn_lab3a.config(bg=COLOR_SECONDARY, fg=COLOR_PRIMARY)
        
        # Clear content area
        self.clear_content()
        
        # Create or show the images view
        if self.images_controller is None:
            # Create a new frame for this view
            view_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
            view_frame.pack(fill='both', expand=True)
            
            # Initialize the controller with this frame
            self.images_controller = AppController(view_frame)
            
            # Store reference to the view's main frame
            self.images_view_frame = view_frame
        else:
            # Show the existing view
            self.images_view_frame.pack(fill='both', expand=True)
        
        self.active_view = "lab3a"

    def clear_content(self):
        """Hides all views from the content area"""
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
