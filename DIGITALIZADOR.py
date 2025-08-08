# DIGITALIZADOR_PRO_V4_Mobile.py
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from io import BytesIO

# Tentativa de importar bibliotecas PDF com fallback
try:
    from pypdf import PdfWriter, PdfReader
except ModuleNotFoundError:
    from PyPDF2 import PdfWriter, PdfReader

# --- CONFIGURAÇÕES GLOBAIS ---
st.set_page_config(page_title="📄 Digitalizador PRO Mobile", layout="centered")
st.markdown("<style>button { font-size: 18px !important; }</style>", unsafe_allow_html=True)

# Dimensões A4 para correção de perspectiva
A4_WIDTH, A4_HEIGHT = 595, 842
MIN_CONTOUR_AREA_RATIO = 0.1

# --- INICIALIZAÇÃO DO ESTADO ---
def initialize_session_state():
    defaults = {
        'document_pages': [],
        'current_image_to_process': None,
        'detected_contour_image': None,
        'processed_image': None,
        'final_image_to_add': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# --- MELHORIA AUTOMÁTICA ---
def enhance_image_for_document(image: Image.Image) -> Image.Image:
    """Melhora qualidade da imagem para smartphone."""
    img = np.array(image.convert('RGB'))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    enhanced_img = Image.fromarray(img)
    enhanced_img.info['dpi'] = (300, 300)
    return enhanced_img

# --- DETECÇÃO DE DOCUMENTO ---
def process_document(image: Image.Image):
    try:
        img_cv = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        img_original = img_cv.copy()

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        doc_corners = None
        image_area = img_cv.shape[0] * img_cv.shape[1]
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (image_area * MIN_CONTOUR_AREA_RATIO):
                doc_corners = approx
                break

        img_with_contour = img_original.copy()
        if doc_corners is not None:
            cv2.drawContours(img_with_contour, [doc_corners], -1, (0, 255, 0), 3)
            st.session_state.detected_contour_image = Image.fromarray(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
            rect = np.array([c[0] for c in doc_corners], dtype="float32")
            s = rect.sum(axis=1)
            ordered_rect = np.zeros((4, 2), dtype="float32")
            ordered_rect[0] = rect[np.argmin(s)]
            ordered_rect[2] = rect[np.argmax(s)]
            diff = np.diff(rect, axis=1)
            ordered_rect[1] = rect[np.argmin(diff)]
            ordered_rect[3] = rect[np.argmax(diff)]

            dst_pts = np.array([[0, 0], [A4_WIDTH, 0], [A4_WIDTH, A4_HEIGHT], [0, A4_HEIGHT]], dtype="float32")
            M = cv2.getPerspectiveTransform(ordered_rect, dst_pts)
            warped = cv2.warpPerspective(img_original, M, (A4_WIDTH, A4_HEIGHT))
            return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        else:
            st.session_state.detected_contour_image = image
            return image
    except Exception as e:
        st.error(f"Erro no processamento: {e}", icon="🚨")
        return None

# --- PÓS-PROCESSAMENTO ---
def apply_post_processing(image: Image.Image, mode: str):
    if mode == 'Otimizado P/B':
        img_gray = ImageOps.grayscale(image)
        img_np = np.array(img_gray)
        thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 15)
        return Image.fromarray(thresh)
    return image

# --- OCR ---
def apply_ocr(image: Image.Image):
    try:
        text = pytesseract.image_to_string(image, lang='por+eng')
        return text if text.strip() else "Nenhum texto detectado."
    except Exception:
        return "Erro no OCR."

# --- PDF ---
def generate_searchable_pdf(image_list: list):
    pdf_writer = PdfWriter()
    for i, image in enumerate(image_list):
        try:
            pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(image.convert("RGB"),
                                                              extension='pdf', lang='por+eng')
            pdf_page_reader = PdfReader(BytesIO(pdf_page_bytes))
            pdf_writer.add_page(pdf_page_reader.pages[0])
        except Exception:
            pdf_buffer = BytesIO()
            image.convert("RGB").save(pdf_buffer, format='PDF', resolution=100.0)
            pdf_buffer.seek(0)
            img_page_reader = PdfReader(pdf_buffer)
            pdf_writer.add_page(img_page_reader.pages[0])
    final_pdf_buffer = BytesIO()
    pdf_writer.write(final_pdf_buffer)
    final_pdf_buffer.seek(0)
    return final_pdf_buffer

# --- INTERFACE ---
st.title("📄 Digitalizador PRO Mobile")

tab1, tab2, tab3 = st.tabs(["📷 Capturar", "📑 Documento", "🔍 OCR"])

with tab1:
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
    camera_photo = st.camera_input("Câmera")
    image_source = camera_photo or uploaded_file

    if image_source and not st.session_state.current_image_to_process:
        st.session_state.current_image_to_process = Image.open(image_source)
        with st.spinner("Processando..."):
            processed = process_document(st.session_state.current_image_to_process)
            enhanced = enhance_image_for_document(processed)
            st.session_state.processed_image = enhanced
            st.session_state.final_image_to_add = enhanced

    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Imagem final", use_column_width=True)
        optimization_mode = st.selectbox("Modo:", ('Cor Original', 'Otimizado P/B'))
        st.session_state.final_image_to_add = apply_post_processing(st.session_state.processed_image, optimization_mode)

        r1, r2 = st.columns(2)
        if r1.button("⟳ 90°"): st.session_state.processed_image = st.session_state.processed_image.rotate(90, expand=True)
        if r2.button("⟲ 90°"): st.session_state.processed_image = st.session_state.processed_image.rotate(-90, expand=True)

        b1, b2 = st.columns(2)
        if b1.button("✅ Adicionar"):
            st.session_state.document_pages.append(st.session_state.final_image_to_add)
            st.session_state.current_image_to_process = None
            st.rerun()
        if b2.button("❌ Descartar"):
            st.session_state.current_image_to_process = None
            st.rerun()

with tab2:
    if not st.session_state.document_pages:
        st.info("Nenhuma página adicionada.")
    else:
        for i, page_img in enumerate(st.session_state.document_pages):
            st.image(page_img, caption=f"Página {i+1}", use_column_width=True)
        filename = st.text_input("Nome do PDF:", "documento.pdf")
        if st.button("📥 Baixar PDF"):
            pdf_buffer = generate_searchable_pdf(st.session_state.document_pages)
            st.download_button("Download", pdf_buffer, file_name=filename, mime="application/pdf")

with tab3:
    if not st.session_state.document_pages:
        st.info("Nenhuma página para OCR.")
    else:
        idx = st.selectbox("Página:", list(range(1, len(st.session_state.document_pages)+1))) - 1
        if st.button("Executar OCR"):
            text = apply_ocr(st.session_state.document_pages[idx])
            st.text_area("Resultado OCR", value=text, height=300)
