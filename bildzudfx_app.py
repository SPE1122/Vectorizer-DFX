"""
Bild-Vektorisierungs-Tool für CNC und Laserschneiden
Streamlit-Version mit OpenCV-Konturerkennung und DXF-Export
Basiert auf der React-Version mit allen Features
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import ezdxf
import io
import tempfile
import os
import re

# Seiten-Konfiguration
st.set_page_config(
    page_title="Bild Vektorisierer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für besseres Design
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stSlider > div > div > div {
        background-color: #4CAF50;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .preset-card {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
    }
    .preset-laser {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 3px solid #FF9800;
    }
    .preset-cnc {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
    }
    .preset-plotting {
        background-color: rgba(156, 39, 176, 0.1);
        border-left: 3px solid #9C27B0;
    }
    .info-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Preset-Definitionen (wie in React-Version)
PRESETS = [
    {
        "id": 1,
        "name": "Feine Details",
        "description": "Für detaillierte Gravuren mit feinen Linien",
        "category": "laser",
        "threshold": 100,
        "smoothing": 1,
        "minSize": 20,
        "icon": "🔥"
    },
    {
        "id": 2,
        "name": "Standard Schnitt",
        "description": "Ausgewogene Einstellungen für normale Schnitte",
        "category": "laser",
        "threshold": 128,
        "smoothing": 2,
        "minSize": 50,
        "icon": "✂️"
    },
    {
        "id": 3,
        "name": "Grobe Konturen",
        "description": "Für große Formen mit wenigen Details",
        "category": "laser",
        "threshold": 150,
        "smoothing": 3,
        "minSize": 100,
        "icon": "📐"
    },
    {
        "id": 4,
        "name": "V-Bit Gravur",
        "description": "Optimiert für V-Fräser Gravuren",
        "category": "cnc",
        "threshold": 120,
        "smoothing": 2,
        "minSize": 30,
        "icon": "⚙️"
    },
    {
        "id": 5,
        "name": "Konturfräsen",
        "description": "Für CNC Außenkonturen",
        "category": "cnc",
        "threshold": 140,
        "smoothing": 3,
        "minSize": 80,
        "icon": "🔧"
    },
    {
        "id": 6,
        "name": "Plotter Zeichnung",
        "description": "Für Stift-Plotter und Zeichenmaschinen",
        "category": "plotting",
        "threshold": 110,
        "smoothing": 2,
        "minSize": 40,
        "icon": "🖊️"
    }
]

# Session State initialisieren
if 'contours' not in st.session_state:
    st.session_state.contours = []
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'dxf_contours' not in st.session_state:
    st.session_state.dxf_contours = []
if 'is_dxf_mode' not in st.session_state:
    st.session_state.is_dxf_mode = False
if 'selected_preset' not in st.session_state:
    st.session_state.selected_preset = None
if 'image_width' not in st.session_state:
    st.session_state.image_width = 800
if 'image_height' not in st.session_state:
    st.session_state.image_height = 600


def apply_preprocessing(image: np.ndarray, brightness: int, contrast: int, 
                        saturation: int, blur: int, sharpen: int) -> np.ndarray:
    """Bildvorverarbeitung mit allen 5 Parametern wie in React-Version"""
    result = image.copy()
    
    # Helligkeit und Kontrast
    if brightness != 0 or contrast != 0:
        alpha = 1.0 + (contrast / 100.0)
        beta = brightness * 2.55
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # Sättigung (HSV-Kanal anpassen)
    if saturation != 0 and len(result.shape) == 3:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + saturation / 100.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Weichzeichnen (Gaussian Blur)
    if blur > 0:
        kernel_size = blur * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # Schärfen (Unsharp Mask)
    if sharpen > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        amount = sharpen / 50.0
        result = cv2.addWeighted(result, 1 + amount, blurred, -amount, 0)
    
    return result


def detect_contours(image: np.ndarray, mode: str, threshold_val: int, 
                   smoothing_val: float, min_size_val: int) -> list:
    """Konturen mit verschiedenen Erkennungsmodi erkennen"""
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if mode == "Graustufen":
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    elif mode == "Kantenerkennung":
        edges = cv2.Canny(gray, threshold_val * 0.5, threshold_val)
        binary = edges
    
    elif mode == "Farbsättigung":
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            _, binary = cv2.threshold(saturation, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    elif mode == "Mittellinie":
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        # Skelettierung für Mittellinie (Zhang-Suen Algorithmus)
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()
        
        max_iterations = 100
        iteration = 0
        while iteration < max_iterations:
            eroded = cv2.erode(temp, element)
            temp_opened = cv2.dilate(eroded, element)
            temp_opened = cv2.subtract(temp, temp_opened)
            skeleton = cv2.bitwise_or(skeleton, temp_opened)
            temp = eroded.copy()
            iteration += 1
            
            if cv2.countNonZero(temp) == 0:
                break
        
        binary = skeleton
    
    else:
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Konturen finden
    contours_cv, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    result_contours = []
    for contour in contours_cv:
        area = cv2.contourArea(contour)
        
        if area < min_size_val:
            continue
        
        # Kontur glätten
        epsilon = smoothing_val * 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:
            points = [(float(p[0][0]), float(p[0][1])) for p in approx]
            result_contours.append({
                'points': points,
                'area': area,
                'perimeter': cv2.arcLength(contour, True),
                'is_closed': True
            })
    
    return result_contours


def chaikin_smoothing(points: list, iterations: int = 1) -> list:
    """Chaikin-Glättungsalgorithmus"""
    if len(points) < 3 or iterations == 0:
        return points
    
    result = points
    for _ in range(iterations):
        new_points = []
        for i in range(len(result) - 1):
            p0 = result[i]
            p1 = result[i + 1]
            
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            
            new_points.append(q)
            new_points.append(r)
        
        result = new_points
    
    return result


def douglas_peucker(points: list, epsilon: float) -> list:
    """Douglas-Peucker Linienvereinfachung"""
    if len(points) < 3 or epsilon <= 0:
        return points
    
    def perpendicular_distance(point, line_start, line_end):
        if line_start == line_end:
            return ((point[0] - line_start[0]) ** 2 + (point[1] - line_start[1]) ** 2) ** 0.5
        
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        t = max(0, min(1, ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx * dx + dy * dy)))
        
        proj_x = line_start[0] + t * dx
        proj_y = line_start[1] + t * dy
        
        return ((point[0] - proj_x) ** 2 + (point[1] - proj_y) ** 2) ** 0.5
    
    max_dist = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_index + 1], epsilon)
        right = douglas_peucker(points[max_index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def merge_nearby_paths(contours: list, tolerance: float) -> list:
    """Pfade verschmelzen, deren Endpunkte nah beieinander liegen"""
    if tolerance <= 0 or len(contours) < 2:
        return contours
    
    open_paths = [c for c in contours if not c.get('is_closed', True)]
    closed_paths = [c for c in contours if c.get('is_closed', True)]
    
    if len(open_paths) < 2:
        return contours
    
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    merged = []
    used = set()
    
    for i, path1 in enumerate(open_paths):
        if i in used:
            continue
        
        current_points = list(path1['points'])
        used.add(i)
        changed = True
        
        while changed:
            changed = False
            for j, path2 in enumerate(open_paths):
                if j in used:
                    continue
                
                p2_start = path2['points'][0]
                p2_end = path2['points'][-1]
                
                if distance(current_points[-1], p2_start) <= tolerance:
                    current_points.extend(path2['points'][1:])
                    used.add(j)
                    changed = True
                elif distance(current_points[-1], p2_end) <= tolerance:
                    current_points.extend(list(reversed(path2['points']))[1:])
                    used.add(j)
                    changed = True
                elif distance(current_points[0], p2_end) <= tolerance:
                    current_points = list(path2['points']) + current_points[1:]
                    used.add(j)
                    changed = True
                elif distance(current_points[0], p2_start) <= tolerance:
                    current_points = list(reversed(path2['points'])) + current_points[1:]
                    used.add(j)
                    changed = True
        
        merged.append({
            **path1,
            'points': current_points
        })
    
    for i, path in enumerate(open_paths):
        if i not in used:
            merged.append(path)
    
    return closed_paths + merged


def apply_post_processing(contours: list, smoothing_passes: int, simplify_mm: float, 
                         merge_mm: float, pixels_per_mm: float = 3.78) -> list:
    """Nachbearbeitung auf Konturen anwenden"""
    result = []
    
    for contour in contours:
        points = list(contour['points'])
        
        # Chaikin-Glättung
        if smoothing_passes > 0:
            points = chaikin_smoothing(points, smoothing_passes)
        
        # Douglas-Peucker Vereinfachung
        if simplify_mm > 0:
            epsilon_px = simplify_mm * pixels_per_mm
            points = douglas_peucker(points, epsilon_px)
        
        if len(points) >= 2:
            result.append({
                **contour,
                'points': points
            })
    
    # Pfadverschmelzung
    if merge_mm > 0:
        merge_px = merge_mm * pixels_per_mm
        result = merge_nearby_paths(result, merge_px)
    
    return result


def draw_contours_on_image(image: np.ndarray, contours: list, 
                          color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Konturen auf Bild zeichnen"""
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        points = contour['points']
        if len(points) >= 2:
            pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
            if contour.get('is_closed', True):
                cv2.polylines(result, [pts], True, color, thickness)
            else:
                cv2.polylines(result, [pts], False, color, thickness)
    
    return result


def export_to_dxf(contours: list, image_width: int, image_height: int) -> bytes:
    """Konturen als DXF exportieren"""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Pixel zu mm (96 DPI)
    scale = 25.4 / 96.0
    
    for contour in contours:
        points = contour['points']
        if len(points) >= 2:
            # Y-Achse invertieren (DXF hat Ursprung unten links)
            dxf_points = [(p[0] * scale, (image_height - p[1]) * scale) for p in points]
            
            if contour.get('is_closed', True) and len(dxf_points) >= 3:
                dxf_points.append(dxf_points[0])
            
            if len(dxf_points) >= 2:
                msp.add_lwpolyline(dxf_points)
    
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        doc.saveas(tmp.name)
        tmp_path = tmp.name
    
    with open(tmp_path, 'rb') as f:
        dxf_bytes = f.read()
    
    os.unlink(tmp_path)
    
    return dxf_bytes


def export_to_svg(contours: list, image_width: int, image_height: int) -> str:
    """Konturen als SVG exportieren"""
    svg_paths = []
    
    for contour in contours:
        points = contour['points']
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
            for p in points[1:]:
                path_d += f" L {p[0]:.2f},{p[1]:.2f}"
            
            if contour.get('is_closed', True):
                path_d += " Z"
            
            svg_paths.append(f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>')
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{image_width}" height="{image_height}" 
     viewBox="0 0 {image_width} {image_height}">
  {chr(10).join(svg_paths)}
</svg>'''
    
    return svg_content


def parse_dxf_file(dxf_content: bytes) -> list:
    """DXF-Datei parsen mit robuster Fehlerbehandlung"""
    contours = []
    
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False, mode='wb') as tmp:
        tmp.write(dxf_content)
        tmp_path = tmp.name
    
    try:
        # Versuche mit ezdxf zu lesen
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        
        for entity in msp:
            try:
                entity_type = entity.dxftype()
                layer_name = entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
                
                if entity_type == 'LWPOLYLINE':
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 2:
                        contours.append({
                            'points': points,
                            'area': 0,
                            'perimeter': 0,
                            'is_closed': entity.closed,
                            'layer': layer_name
                        })
                
                elif entity_type == 'LINE':
                    start = (entity.dxf.start.x, entity.dxf.start.y)
                    end = (entity.dxf.end.x, entity.dxf.end.y)
                    contours.append({
                        'points': [start, end],
                        'area': 0,
                        'perimeter': 0,
                        'is_closed': False,
                        'layer': layer_name
                    })
                
                elif entity_type == 'CIRCLE':
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    points = []
                    for i in range(64):
                        angle = 2 * np.pi * i / 64
                        points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
                    contours.append({
                        'points': points,
                        'area': np.pi * r * r,
                        'perimeter': 2 * np.pi * r,
                        'is_closed': True,
                        'layer': layer_name
                    })
                
                elif entity_type == 'ARC':
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    start_angle = np.radians(entity.dxf.start_angle)
                    end_angle = np.radians(entity.dxf.end_angle)
                    
                    if end_angle < start_angle:
                        end_angle += 2 * np.pi
                    
                    points = []
                    steps = 32
                    for i in range(steps + 1):
                        angle = start_angle + (end_angle - start_angle) * i / steps
                        points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
                    
                    contours.append({
                        'points': points,
                        'area': 0,
                        'perimeter': 0,
                        'is_closed': False,
                        'layer': layer_name
                    })
                
                elif entity_type == 'POLYLINE':
                    points = []
                    for vertex in entity.vertices:
                        if hasattr(vertex.dxf, 'location'):
                            points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                    if len(points) >= 2:
                        contours.append({
                            'points': points,
                            'area': 0,
                            'perimeter': 0,
                            'is_closed': entity.is_closed,
                            'layer': layer_name
                        })
                
                elif entity_type == 'SPLINE':
                    try:
                        if hasattr(entity, 'control_points'):
                            points = [(p[0], p[1]) for p in entity.control_points]
                            if len(points) >= 2:
                                contours.append({
                                    'points': points,
                                    'area': 0,
                                    'perimeter': 0,
                                    'is_closed': entity.closed if hasattr(entity, 'closed') else False,
                                    'layer': layer_name
                                })
                    except Exception:
                        pass
                
                elif entity_type == 'ELLIPSE':
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    major = entity.dxf.major_axis
                    ratio = entity.dxf.ratio
                    
                    a = (major.x ** 2 + major.y ** 2) ** 0.5
                    b = a * ratio
                    rotation = np.arctan2(major.y, major.x)
                    
                    points = []
                    for i in range(64):
                        angle = 2 * np.pi * i / 64
                        x = a * np.cos(angle)
                        y = b * np.sin(angle)
                        rx = x * np.cos(rotation) - y * np.sin(rotation) + cx
                        ry = x * np.sin(rotation) + y * np.cos(rotation) + cy
                        points.append((rx, ry))
                    
                    contours.append({
                        'points': points,
                        'area': np.pi * a * b,
                        'perimeter': 0,
                        'is_closed': True,
                        'layer': layer_name
                    })
                    
            except Exception as e:
                # Einzelne Entity-Fehler ignorieren, weiter parsen
                continue
    
    except ezdxf.DXFStructureError as e:
        # Bei Struktur-Fehlern versuchen wir einen einfachen Text-Parser
        contours = parse_dxf_simple(dxf_content)
    
    except Exception as e:
        # Bei anderen Fehlern auch den einfachen Parser versuchen
        contours = parse_dxf_simple(dxf_content)
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return contours


def parse_dxf_simple(dxf_content: bytes) -> list:
    """Einfacher fallback DXF-Parser für problematische Dateien"""
    contours = []
    
    try:
        text = dxf_content.decode('utf-8', errors='ignore')
    except:
        text = dxf_content.decode('latin-1', errors='ignore')
    
    lines = text.split('\n')
    
    i = 0
    current_entity = None
    current_data = {}
    
    while i < len(lines) - 1:
        try:
            code = int(lines[i].strip())
            value = lines[i + 1].strip()
            
            if code == 0:
                # Neuer Entity oder Sektion
                if current_entity == 'LINE' and '10' in current_data and '20' in current_data:
                    try:
                        x1 = float(current_data.get('10', 0))
                        y1 = float(current_data.get('20', 0))
                        x2 = float(current_data.get('11', 0))
                        y2 = float(current_data.get('21', 0))
                        contours.append({
                            'points': [(x1, y1), (x2, y2)],
                            'area': 0,
                            'perimeter': 0,
                            'is_closed': False,
                            'layer': current_data.get('8', '0')
                        })
                    except:
                        pass
                
                current_entity = value
                current_data = {}
            
            else:
                current_data[str(code)] = value
            
            i += 2
            
        except ValueError:
            i += 1
    
    return contours


def normalize_dxf_contours(contours: list, target_width: int = 800, 
                          target_height: int = 600) -> tuple:
    """DXF-Konturen auf Canvas normalisieren"""
    if not contours:
        return contours, 1.0, 0, 0
    
    all_points = []
    for c in contours:
        all_points.extend(c['points'])
    
    if not all_points:
        return contours, 1.0, 0, 0
    
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y) * 0.9
    
    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (target_width - scaled_width) / 2 - min_x * scale
    offset_y = (target_height - scaled_height) / 2 - min_y * scale
    
    normalized = []
    for contour in contours:
        new_points = []
        for p in contour['points']:
            x = p[0] * scale + offset_x
            y = target_height - (p[1] * scale + offset_y)
            new_points.append((x, y))
        normalized.append({
            **contour,
            'points': new_points
        })
    
    return normalized, scale, offset_x, offset_y


# ============== HAUPTANWENDUNG ==============

# Titel mit Icon
st.title("✏️ Bild Vektorisierer")
st.markdown("*Konvertiert Bilder zu Vektorkonturen für CNC und Laserschneiden*")

# Sidebar
with st.sidebar:
    # Presets Sektion
    st.header("🎯 Presets")
    st.caption("Schnellkonfigurationen für typische Anwendungen")
    
    # Preset nach Kategorie gruppieren
    categories = {"laser": "🔥 Laserschneiden", "cnc": "⚙️ CNC Fräsen", "plotting": "🖊️ Plotten"}
    
    for cat_key, cat_name in categories.items():
        with st.expander(cat_name, expanded=False):
            cat_presets = [p for p in PRESETS if p["category"] == cat_key]
            for preset in cat_presets:
                is_selected = st.session_state.selected_preset == preset["id"]
                button_type = "primary" if is_selected else "secondary"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{preset['icon']} {preset['name']}", 
                        key=f"preset_{preset['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        if is_selected:
                            st.session_state.selected_preset = None
                        else:
                            st.session_state.selected_preset = preset["id"]
                            st.rerun()
                
                with col2:
                    if is_selected:
                        st.markdown("✓")
                
                if is_selected:
                    st.caption(preset["description"])
    
    st.divider()
    
    # Erkennungsmodus
    st.header("🔍 Erkennung")
    detection_mode = st.selectbox(
        "Erkennungsmodus",
        options=["Graustufen", "Kantenerkennung", "Farbsättigung", "Mittellinie"],
        index=1,
        help="Graustufen: B/W Bilder\nKantenerkennung: Farbige Bilder\nFarbsättigung: Lebhafte Farben\nMittellinie: CNC V-Fräser"
    )
    
    # Parameter aus Preset oder manuell
    selected_preset_data = next((p for p in PRESETS if p["id"] == st.session_state.selected_preset), None)
    
    default_threshold = selected_preset_data["threshold"] if selected_preset_data else 128
    default_smoothing = selected_preset_data["smoothing"] if selected_preset_data else 2
    default_min_size = selected_preset_data["minSize"] if selected_preset_data else 50
    
    threshold = st.slider(
        "Schwellwert",
        min_value=0,
        max_value=255,
        value=default_threshold,
        help="Bestimmt, welche Pixel als Kontur erkannt werden"
    )
    
    smoothing = st.slider(
        "Glättung",
        min_value=0.0,
        max_value=5.0,
        value=float(default_smoothing),
        step=0.5,
        help="Epsilon für Konturen-Approximation"
    )
    
    min_size = st.slider(
        "Mindestgröße (px²)",
        min_value=0,
        max_value=1000,
        value=default_min_size,
        step=10,
        help="Konturen kleiner als dieser Wert werden ignoriert"
    )
    
    st.divider()
    
    # Bildbearbeitung (alle 5 Parameter)
    st.header("🖼️ Bildbearbeitung")
    
    brightness = st.slider("Helligkeit", -100, 100, 0, help="Bildhelligkeit anpassen")
    contrast = st.slider("Kontrast", -100, 100, 0, help="Bildkontrast anpassen")
    saturation = st.slider("Sättigung", -100, 100, 0, help="Farbsättigung anpassen")
    blur_amount = st.slider("Weichzeichnen", 0, 10, 0, help="Gaussian Blur anwenden")
    sharpen_amount = st.slider("Schärfen", 0, 100, 0, help="Unsharp Mask zum Schärfen")
    
    if st.button("🔄 Bildbearbeitung zurücksetzen", use_container_width=True):
        st.rerun()
    
    st.divider()
    
    # DXF Nachbearbeitung
    st.header("📐 DXF Nachbearbeitung")
    
    additional_smoothing = st.slider(
        "Zusatzglättung",
        min_value=0,
        max_value=5,
        value=0,
        help="Chaikin-Glättungsdurchgänge für sanftere CNC-Bewegungen"
    )
    
    simplify_tolerance = st.slider(
        "Vereinfachung (mm)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Douglas-Peucker Vereinfachung reduziert Punktanzahl"
    )
    
    merge_tolerance = st.slider(
        "Pfadverschmelzung (mm)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Endpunkte näher als dieser Wert werden verbunden"
    )
    
    show_original = st.checkbox("Original anzeigen", value=False, 
                                help="Zeigt die Original-Konturen ohne Nachbearbeitung")


# Hauptbereich mit zwei Spalten
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Datei hochladen")
    
    uploaded_file = st.file_uploader(
        "Bild oder DXF-Datei wählen",
        type=['png', 'jpg', 'jpeg', 'bmp', 'dxf'],
        help="PNG, JPG, BMP oder DXF"
    )
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'dxf':
            # DXF-Modus
            st.session_state.is_dxf_mode = True
            dxf_content = uploaded_file.read()
            
            try:
                raw_contours = parse_dxf_file(dxf_content)
                
                if len(raw_contours) == 0:
                    st.warning("⚠️ Keine Konturen in der DXF-Datei gefunden")
                else:
                    normalized, scale, ox, oy = normalize_dxf_contours(raw_contours)
                    st.session_state.dxf_contours = normalized
                    st.session_state.contours = normalized
                    st.session_state.image_width = 800
                    st.session_state.image_height = 600
                    
                    st.success(f"✅ DXF importiert: {len(normalized)} Konturen")
                    
                    # Info über Entities
                    layers = set(c.get('layer', '0') for c in normalized)
                    st.info(f"📊 Layer: {', '.join(layers)}")
                    
                    # Leeres Bild für Vorschau erstellen
                    preview_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
                    st.session_state.original_image = preview_img
                    st.session_state.processed_image = preview_img
                    
            except Exception as e:
                st.error(f"❌ DXF-Fehler: {str(e)}")
        
        else:
            # Bild-Modus
            st.session_state.is_dxf_mode = False
            
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Format konvertieren
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Große Bilder herunterskalieren
            max_dim = 1200
            h, w = image_np.shape[:2]
            if max(h, w) > max_dim:
                scale_factor = max_dim / max(h, w)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                image_np = cv2.resize(image_np, (new_w, new_h))
                st.info(f"📏 Bild skaliert: {w}x{h} → {new_w}x{new_h}")
            
            st.session_state.original_image = image_np
            st.session_state.image_width = image_np.shape[1]
            st.session_state.image_height = image_np.shape[0]
            
            # Vorverarbeitung anwenden
            processed = apply_preprocessing(
                image_np, brightness, contrast, saturation, blur_amount, sharpen_amount
            )
            st.session_state.processed_image = processed
            
            # Konturen erkennen
            contours = detect_contours(
                processed,
                detection_mode,
                threshold,
                smoothing,
                min_size
            )
            
            st.session_state.contours = contours
            st.success(f"✅ {len(contours)} Konturen erkannt")
            
            # Download für bearbeitetes Bild
            if brightness != 0 or contrast != 0 or saturation != 0 or blur_amount > 0 or sharpen_amount > 0:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                processed_pil = Image.fromarray(processed_rgb)
                img_buffer = io.BytesIO()
                processed_pil.save(img_buffer, format='PNG')
                
                st.download_button(
                    label="📥 Bearbeitetes Bild herunterladen",
                    data=img_buffer.getvalue(),
                    file_name="bearbeitet.png",
                    mime="image/png"
                )

with col2:
    st.subheader("👁️ Vorschau")
    
    if st.session_state.original_image is not None:
        # Konturen mit Nachbearbeitung
        contours_to_display = st.session_state.contours
        
        if not show_original and (additional_smoothing > 0 or simplify_tolerance > 0 or merge_tolerance > 0):
            contours_to_display = apply_post_processing(
                contours_to_display,
                additional_smoothing,
                simplify_tolerance,
                merge_tolerance
            )
        
        # Bild mit Konturen zeichnen
        if st.session_state.is_dxf_mode:
            # Weißer Hintergrund für DXF
            display_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        else:
            display_image = st.session_state.processed_image.copy()
        
        # Konturen zeichnen
        contour_image = draw_contours_on_image(
            display_image, 
            contours_to_display, 
            color=(0, 200, 0), 
            thickness=2
        )
        
        # BGR zu RGB für Anzeige
        display_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        st.image(display_rgb, use_container_width=True)
        
        # Statistiken
        total_points = sum(len(c['points']) for c in contours_to_display)
        st.markdown(f"""
        **Statistik:** {len(contours_to_display)} Konturen, {total_points} Punkte | 
        **Größe:** {st.session_state.image_width} × {st.session_state.image_height} px
        """)
        
        # Export-Buttons
        st.subheader("📤 Export")
        
        col_dxf, col_svg = st.columns(2)
        
        with col_dxf:
            dxf_data = export_to_dxf(
                contours_to_display,
                st.session_state.image_width,
                st.session_state.image_height
            )
            
            st.download_button(
                label="📥 DXF herunterladen",
                data=dxf_data,
                file_name="export.dxf",
                mime="application/dxf",
                use_container_width=True
            )
        
        with col_svg:
            svg_data = export_to_svg(
                contours_to_display,
                st.session_state.image_width,
                st.session_state.image_height
            )
            
            st.download_button(
                label="📥 SVG herunterladen",
                data=svg_data,
                file_name="export.svg",
                mime="image/svg+xml",
                use_container_width=True
            )
    
    else:
        # Platzhalter wenn kein Bild geladen
        st.markdown("""
        <div style="
            background-color: #f5f5f5;
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 60px;
            text-align: center;
            color: #888;
        ">
            <p style="font-size: 48px; margin: 0;">📷</p>
            <p style="font-size: 18px; margin-top: 20px;">Bild oder DXF-Datei hochladen</p>
            <p style="font-size: 14px;">Unterstützt: PNG, JPG, BMP, DXF</p>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    Bild Vektorisierer v2.0 | Für CNC, Laserschneiden und Plotten
</div>
""", unsafe_allow_html=True)
