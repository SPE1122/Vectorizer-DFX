"""
Bild-Vektorisierungs-Tool für CNC und Laserschneiden
Streamlit-Version mit OpenCV-Konturerkennung und DXF-Export
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import ezdxf
import io
import tempfile
import os

st.set_page_config(
    page_title="Bild Vektorisierer",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Bild Vektorisierer")
st.markdown("*Konvertiert Bilder zu Vektorkonturen für CNC und Laserschneiden*")

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

# Sidebar mit Parametern
with st.sidebar:
    st.header("⚙️ Vektorisierungs-Parameter")
    
    detection_mode = st.selectbox(
        "Erkennungsmodus",
        options=["Graustufen", "Kantenerkennung", "Farbsättigung", "Mittellinie"],
        index=1,
        help="Graustufen: B/W Bilder\nKantenerkennung: Farbige Bilder\nFarbsättigung: Lebhafte Farben\nMittellinie: CNC V-Fräser"
    )
    
    threshold = st.slider(
        "Schwellwert",
        min_value=0,
        max_value=255,
        value=128,
        help="Bestimmt, welche Pixel als Kontur erkannt werden"
    )
    
    smoothing = st.slider(
        "Glättung",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Epsilon für Konturen-Approximation"
    )
    
    min_size = st.slider(
        "Mindestgröße (px²)",
        min_value=0,
        max_value=1000,
        value=100,
        step=10,
        help="Konturen kleiner als dieser Wert werden ignoriert"
    )
    
    st.divider()
    st.header("🖼️ Bildbearbeitung")
    
    brightness = st.slider("Helligkeit", -100, 100, 0)
    contrast = st.slider("Kontrast", -100, 100, 0)
    blur_amount = st.slider("Weichzeichnen", 0, 10, 0)
    
    st.divider()
    st.header("📐 DXF Nachbearbeitung")
    
    additional_smoothing = st.slider(
        "Zusatzglättung",
        min_value=0,
        max_value=5,
        value=0,
        help="Chaikin-Glättungsdurchgänge"
    )
    
    simplify_tolerance = st.slider(
        "Vereinfachung (mm)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Douglas-Peucker Vereinfachung"
    )
    
    merge_tolerance = st.slider(
        "Pfadverschmelzung (mm)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Endpunkte näher als dieser Wert werden verbunden"
    )


def apply_preprocessing(image: np.ndarray, brightness: int, contrast: int, blur: int) -> np.ndarray:
    """Bildvorverarbeitung anwenden"""
    result = image.copy()
    
    # Helligkeit und Kontrast
    if brightness != 0 or contrast != 0:
        alpha = 1.0 + (contrast / 100.0)
        beta = brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # Weichzeichnen
    if blur > 0:
        kernel_size = blur * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    return result


def detect_contours(image: np.ndarray, mode: str, threshold_val: int, smoothing_val: float, min_size_val: int) -> list:
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
        # Skelettierung für Mittellinie
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()
        
        while True:
            eroded = cv2.erode(temp, element)
            temp_opened = cv2.dilate(eroded, element)
            temp_opened = cv2.subtract(temp, temp_opened)
            skeleton = cv2.bitwise_or(skeleton, temp_opened)
            temp = eroded.copy()
            
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
    
    # Nur offene Pfade verschmelzen
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
                
                # Prüfen ob Endpunkte nah genug sind
                if distance(current_points[-1], p2_start) <= tolerance:
                    current_points.extend(path2['points'][1:])
                    used.add(j)
                    changed = True
                elif distance(current_points[-1], p2_end) <= tolerance:
                    current_points.extend(reversed(path2['points'][:-1]))
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
    
    # Nicht verwendete offene Pfade hinzufügen
    for i, path in enumerate(open_paths):
        if i not in used:
            merged.append(path)
    
    return closed_paths + merged


def apply_post_processing(contours: list, smoothing_passes: int, simplify_mm: float, merge_mm: float, pixels_per_mm: float = 3.78) -> list:
    """Nachbearbeitung auf Konturen anwenden"""
    result = []
    
    for contour in contours:
        points = contour['points']
        
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


def draw_contours_on_image(image: np.ndarray, contours: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
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
                dxf_points.append(dxf_points[0])  # Schließen
            
            if len(dxf_points) >= 2:
                msp.add_lwpolyline(dxf_points)
    
    # In Bytes exportieren
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
    """DXF-Datei parsen und Konturen extrahieren"""
    contours = []
    
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(dxf_content)
        tmp_path = tmp.name
    
    try:
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in entity.get_points()]
                if len(points) >= 2:
                    contours.append({
                        'points': points,
                        'area': 0,
                        'perimeter': 0,
                        'is_closed': entity.closed,
                        'layer': entity.dxf.layer
                    })
            
            elif entity.dxftype() == 'LINE':
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                contours.append({
                    'points': [start, end],
                    'area': 0,
                    'perimeter': 0,
                    'is_closed': False,
                    'layer': entity.dxf.layer
                })
            
            elif entity.dxftype() == 'CIRCLE':
                cx, cy = entity.dxf.center.x, entity.dxf.center.y
                r = entity.dxf.radius
                # Kreis als Polygon approximieren
                points = []
                for i in range(64):
                    angle = 2 * np.pi * i / 64
                    points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
                contours.append({
                    'points': points,
                    'area': np.pi * r * r,
                    'perimeter': 2 * np.pi * r,
                    'is_closed': True,
                    'layer': entity.dxf.layer
                })
            
            elif entity.dxftype() == 'ARC':
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
                    'layer': entity.dxf.layer
                })
            
            elif entity.dxftype() == 'POLYLINE':
                points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                if len(points) >= 2:
                    contours.append({
                        'points': points,
                        'area': 0,
                        'perimeter': 0,
                        'is_closed': entity.is_closed,
                        'layer': entity.dxf.layer
                    })
            
            elif entity.dxftype() == 'SPLINE':
                try:
                    points = [(p[0], p[1]) for p in entity.control_points]
                    if len(points) >= 2:
                        contours.append({
                            'points': points,
                            'area': 0,
                            'perimeter': 0,
                            'is_closed': entity.closed,
                            'layer': entity.dxf.layer
                        })
                except:
                    pass
            
            elif entity.dxftype() == 'ELLIPSE':
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
                    # Rotation anwenden
                    rx = x * np.cos(rotation) - y * np.sin(rotation) + cx
                    ry = x * np.sin(rotation) + y * np.cos(rotation) + cy
                    points.append((rx, ry))
                
                contours.append({
                    'points': points,
                    'area': np.pi * a * b,
                    'perimeter': 0,
                    'is_closed': True,
                    'layer': entity.dxf.layer
                })
    
    finally:
        os.unlink(tmp_path)
    
    return contours


def normalize_dxf_contours(contours: list, target_width: int = 800, target_height: int = 600) -> tuple:
    """DXF-Konturen auf Canvas normalisieren"""
    if not contours:
        return contours, 1.0, 0, 0
    
    # Bounds berechnen
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


# Hauptbereich
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
                normalized, scale, ox, oy = normalize_dxf_contours(raw_contours)
                st.session_state.dxf_contours = normalized
                st.session_state.contours = normalized
                
                st.success(f"✅ DXF importiert: {len(normalized)} Konturen")
                
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
            
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            st.session_state.original_image = image_np
            
            # Vorverarbeitung
            processed = apply_preprocessing(image_np, brightness, contrast, blur_amount)
            st.session_state.processed_image = processed
            
            # Konturen erkennen
            contours = detect_contours(
                processed,
                detection_mode,
                threshold,
                smoothing,
                min_size
            )
            
            # Nachbearbeitung
            if additional_smoothing > 0 or simplify_tolerance > 0 or merge_tolerance > 0:
                contours = apply_post_processing(
                    contours,
                    additional_smoothing,
                    simplify_tolerance,
                    merge_tolerance
                )
            
            st.session_state.contours = contours
            st.success(f"✅ {len(contours)} Konturen erkannt")
    
    # Export-Buttons
    if st.session_state.contours:
        st.divider()
        st.subheader("💾 Export")
        
        col_dxf, col_svg = st.columns(2)
        
        with col_dxf:
            if st.button("📐 DXF Export", use_container_width=True):
                img = st.session_state.original_image
                if img is not None:
                    h, w = img.shape[:2]
                    dxf_bytes = export_to_dxf(st.session_state.contours, w, h)
                    st.download_button(
                        "⬇️ DXF herunterladen",
                        data=dxf_bytes,
                        file_name="konturen.dxf",
                        mime="application/dxf",
                        use_container_width=True
                    )
        
        with col_svg:
            if st.button("🎨 SVG Export", use_container_width=True):
                img = st.session_state.original_image
                if img is not None:
                    h, w = img.shape[:2]
                    svg_content = export_to_svg(st.session_state.contours, w, h)
                    st.download_button(
                        "⬇️ SVG herunterladen",
                        data=svg_content,
                        file_name="konturen.svg",
                        mime="image/svg+xml",
                        use_container_width=True
                    )

with col2:
    st.subheader("👁️ Vorschau")
    
    show_overlay = st.checkbox("Konturen anzeigen", value=True)
    
    if st.session_state.original_image is not None:
        # Vorschaubild erstellen
        preview = st.session_state.processed_image.copy() if st.session_state.processed_image is not None else st.session_state.original_image.copy()
        
        if show_overlay and st.session_state.contours:
            # Konturen zeichnen
            preview = draw_contours_on_image(preview, st.session_state.contours, color=(0, 200, 0), thickness=2)
        
        # Von BGR zu RGB für Streamlit
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, use_container_width=True)
        
        # Statistiken
        if st.session_state.contours:
            st.info(f"📊 **{len(st.session_state.contours)}** Konturen | "
                   f"**{sum(len(c['points']) for c in st.session_state.contours)}** Punkte gesamt")
    else:
        st.info("⬆️ Bitte laden Sie ein Bild oder eine DXF-Datei hoch")

# Footer
st.divider()
st.caption("Bild Vektorisierer • Streamlit Version • Für CNC und Laserschneiden")
