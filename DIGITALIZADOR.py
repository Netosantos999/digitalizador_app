# DIGITALIZADOR_PRO_V2.py
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- CONFIGURAÇÕES GLOBAIS E CONSTANTES ---
st.set_page_config(page_title="Digitalizador PRO", layout="wide")

# Constantes para dimensões e processamento
A4_WIDTH, A4_HEIGHT = 595, 842
MIN_CONTOUR_AREA_RATIO = 0.15 # Área mínima para ser considerado um documento

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'document_pages' not in st.session_state:
    st.session_state.document_pages = []
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'detected_contour_image' not in st.session_state:
    st.session_state.detected_contour_image = None

# --- FUNÇÕES DE PROCESSAMENTO DE IMAGEM ---

def process_document(image: Image.Image):
    """
    Detecta as bordas de um documento, desenha o contorno para feedback
    e retorna a imagem com perspectiva corrigida.
    """
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
    else:
        st.session_state.detected_contour_image = image
        st.warning("Não foi possível detectar um documento. A imagem original será usada.")
        return image

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

def apply_post_processing(image: Image.Image, mode: str):
    """Aplica filtros de otimização na imagem."""
    if mode == 'Otimizado para Leitura (P&B)':
        img_gray = ImageOps.grayscale(image)
        img_np = np.array(img_gray)
        thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        return Image.fromarray(thresh)
    return image

# --- FUNÇÕES AUXILIARES (OCR E PDF) ---

def apply_ocr(image: Image.Image):
    """Extrai texto da imagem usando Tesseract."""
    try:
        text = pytesseract.image_to_string(image, lang='por+eng')
        return text if text.strip() else "Nenhum texto detectado."
    except Exception:
        st.error("Erro no OCR. Verifique se o Tesseract está instalado e acessível no PATH do sistema.")
        return ""

def generate_pdf_from_images(image_list: list):
    """Gera um PDF multi-páginas a partir de uma lista de imagens."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    for image in image_list:
        image_reader = ImageReader(image)
        img_width, img_height = image.size
        aspect = img_height / float(img_width)
        
        draw_width = page_width * 0.9
        draw_height = draw_width * aspect
        
        if draw_height > page_height * 0.9:
            draw_height = page_height * 0.9
            draw_width = draw_height / aspect

        x = (page_width - draw_width) / 2
        y = (page_height - draw_height) / 2

        c.drawImage(image_reader, x, y, width=draw_width, height=draw_height, preserveAspectRatio=True, mask='auto')
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("📄 Digitalizador PRO V2")

# <--- MELHORIA: Instruções mais completas, incluindo como escolher o local do download.
with st.expander("💡 Como obter os melhores resultados?"):
    st.markdown("""
    - **Fundo Contrastante:** Coloque o papel sobre uma mesa escura.
    - **Boa Iluminação:** Evite sombras sobre o documento.
    - **Enquadramento:** Tente capturar apenas o documento e um pouco do fundo.
    - **Escolher Onde Salvar:** Para que o navegador pergunte onde você quer salvar cada arquivo, ative a opção correspondente nas configurações de Downloads do seu navegador (Chrome, Firefox, Edge, etc.).
    """)

tab1, tab2, tab3 = st.tabs(["1. Digitalizar e Processar", "2. Montar Documento", "3. Extrair Texto (OCR)"])

with tab1:
    st.header("Passo 1: Capturar Imagem")
    image_source = None
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload de arquivo", type=["png", "jpg", "jpeg"], key="uploader")
    with col2:
        camera_photo = st.camera_input("Usar a câmera", key="camera")

    if camera_photo: image_source = camera_photo
    elif uploaded_file: image_source = uploaded_file

    if image_source:
        st.session_state.original_image = Image.open(image_source)
        
        with st.spinner("Analisando imagem e detectando bordas..."):
            st.session_state.processed_image = process_document(st.session_state.original_image)

        st.header("Passo 2: Verificar e Otimizar")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Detecção de Bordas")
            st.image(st.session_state.detected_contour_image, caption="Contorno detectado na imagem original.", use_column_width=True)
        with col_b:
            st.subheader("Resultado Processado")
            
            optimization_mode = st.selectbox("Modo de Otimização:", ('Cor Original', 'Otimizado para Leitura (P&B)'), key="optimization")
            final_image = apply_post_processing(st.session_state.processed_image, optimization_mode)
            st.image(final_image, caption="Imagem final ajustada.", use_column_width=True)

        if st.button("✅ Adicionar ao Documento", use_container_width=True, type="primary"):
            st.session_state.document_pages.append(final_image)
            st.success(f"Página {len(st.session_state.document_pages)} adicionada! Vá para a aba 'Montar Documento'.")

with tab2:
    st.header("Documento Atual")
    if not st.session_state.document_pages:
        st.info("Nenhuma página foi adicionada ao documento ainda. Volte para a primeira aba para digitalizar.")
    else:
        st.subheader(f"Páginas Adicionadas: {len(st.session_state.document_pages)}")
        
        cols = st.columns(5)
        for i, page_img in enumerate(st.session_state.document_pages):
            col = cols[i % 5]
            with col:
                st.image(page_img, caption=f"Página {i+1}", use_column_width=True)
                if st.button(f"Remover", key=f"del_{i}", use_container_width=True):
                    st.session_state.document_pages.pop(i)
                    st.rerun()
        
        st.divider()
        st.subheader("Finalizar")

        # <--- MELHORIA: Campo para o usuário definir o nome do arquivo.
        default_filename = f"documento_{st.session_state.document_pages[0].getbands()}_{len(st.session_state.document_pages)}pags.pdf"
        file_name_input = st.text_input("Nome do arquivo para download:", default_filename)
        
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            pdf_buffer = generate_pdf_from_images(st.session_state.document_pages)
            
            # Garante que o nome do arquivo tenha a extensão .pdf
            if file_name_input and not file_name_input.lower().endswith('.pdf'):
                file_name_input += '.pdf'
                
            st.download_button(
                label="📥 Baixar PDF Final",
                data=pdf_buffer,
                file_name=file_name_input, # Usa o nome de arquivo dinâmico
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        with action_col2:
            if st.button("🗑️ Limpar Tudo", use_container_width=True):
                st.session_state.document_pages = []
                st.rerun()

with tab3:
    st.header("Extração de Texto (OCR)")
    if not st.session_state.document_pages:
        st.info("Adicione páginas ao documento na primeira aba para poder extrair texto.")
    else:
        page_options = {f"Página {i+1}": i for i, _ in enumerate(st.session_state.document_pages)}
        selected_page_label = st.selectbox("Selecione a página para extrair o texto:", options=page_options.keys())
        
        if selected_page_label:
            page_index = page_options[selected_page_label]
            selected_image = st.session_state.document_pages[page_index]
            
            col_ocr1, col_ocr2 = st.columns(2)
            with col_ocr1:
                st.subheader("Imagem Selecionada")
                st.image(selected_image, use_column_width=True)
            with col_ocr2:
                st.subheader("Texto Extraído")
                if st.button("Processar OCR", use_container_width=True):
                    with st.spinner("Lendo texto da imagem..."):
                        extracted_text = apply_ocr(selected_image)
                        st.text_area("Resultado:", value=extracted_text, height=300)