import os
import re
import numpy as np
from PIL import Image

def normalize_image(image_path, size=(48, 48)):
    """Normaliza una imagen para el procesamiento"""
    image = Image.open(image_path).convert('RGB').resize(size)
    r, g, b = image.split()
    r_data = np.asarray(r, dtype=np.float32) / 255.0
    g_data = np.asarray(g, dtype=np.float32) / 255.0
    b_data = np.asarray(b, dtype=np.float32) / 255.0
    return r_data.flatten(), g_data.flatten(), b_data.flatten()

def save_normalized_data(directory, output_file):
    """Guarda datos normalizados de imágenes en un archivo de texto"""
    with open(output_file, 'w') as f:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                vocal = filename[0].upper()
                if vocal not in ['A', 'E', 'I', 'O', 'U']:
                    continue
                label = {
                    'A': [1, 0, 0, 0, 0],
                    'E': [0, 1, 0, 0, 0],
                    'I': [0, 0, 1, 0, 0],
                    'O': [0, 0, 0, 1, 0],
                    'U': [0, 0, 0, 0, 1],
                }[vocal]
                path = os.path.join(directory, filename)
                r, g, b = normalize_image(path)
                r_str = [float(x) for x in r]
                g_str = [float(x) for x in g]
                b_str = [float(x) for x in b]
                f.write(f"R{r_str}+G{g_str}+B{b_str}:{label}\n")
    print(f"Imágenes normalizadas y guardadas en {output_file}")

def load_training_data(txt_file):
    """Carga datos de entrenamiento desde un archivo de texto"""
    inputs, outputs = [], []
    pattern = r'R\[(.*?)\]\+G\[(.*?)\]\+B\[(.*?)\]:(\[.*\])'
    with open(txt_file, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                r_vals = list(map(float, match.group(1).split(',')))
                g_vals = list(map(float, match.group(2).split(',')))
                b_vals = list(map(float, match.group(3).split(',')))
                label = eval(match.group(4))
                input_vec = r_vals + g_vals + b_vals
                inputs.append(input_vec)
                outputs.append(label)
            else:
                print("⚠️ Línea con formato inválido:", line.strip())
    return np.array(inputs), np.array(outputs)

def determine_dominant_color(r, g, b):
    """
    Determina el color dominante en una imagen y calcula los porcentajes de cada color
    
    Args:
        r: Array de valores de rojo normalizados
        g: Array de valores de verde normalizados
        b: Array de valores de azul normalizados
        
    Returns:
        tuple: (color_dominante, porcentajes)
            - color_dominante: String con el nombre del color dominante ("Rojo", "Verde", "Azul")
            - porcentajes: Diccionario con los porcentajes de cada color {"Rojo": xx.x, "Verde": yy.y, "Azul": zz.z}
    """
    # Calcular la suma de cada componente de color
    r_sum = np.sum(r)
    g_sum = np.sum(g)
    b_sum = np.sum(b)
    
    # Calcular el total para obtener porcentajes
    total = r_sum + g_sum + b_sum
    
    # Evitar división por cero
    if total == 0:
        total = 1.0
    
    # Calcular porcentajes (0-100)
    r_percent = (r_sum / total) * 100
    g_percent = (g_sum / total) * 100
    b_percent = (b_sum / total) * 100
    
    # Crear diccionario de porcentajes
    porcentajes = {
        "Rojo": r_percent,
        "Verde": g_percent,
        "Azul": b_percent
    }
    
    # Determinar el color dominante
    if g_sum > r_sum and g_sum > b_sum:
        color_dominante = "Verde"
    elif b_sum > r_sum and b_sum > g_sum:
        color_dominante = "Azul"
    else:
        color_dominante = "Rojo"
    
    return color_dominante, porcentajes

def process_test_image(image_path):
    """
    Procesa una imagen de prueba y devuelve su vector de características, 
    color dominante y porcentajes de cada color
    """
    r, g, b = normalize_image(image_path)
    input_vec = np.concatenate([r, g, b])
    color_dominante, porcentajes = determine_dominant_color(r, g, b)
    return input_vec, color_dominante, porcentajes