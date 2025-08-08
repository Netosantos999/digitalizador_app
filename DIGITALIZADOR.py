# DIGITALIZADOR_PRO_V3.1.py
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from io import BytesIO
from pypdf import PdfWriter, PdfReader

# --- CONFIGURAÇÕES GLOBAIS E CONSTANTES ---
st.set_page_config(page_title="Digitalizador PRO V3", layout="wide")

# Constantes para dimensões e processamento
A4_WIDTH, A4_HEIGHT = 595, 842
MIN_CONTOUR_AREA_RATIO = 0.1 # Área mínima para ser considerado um documento

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
def initialize_session_state():
    """Inicializa as variáveis de estado da sessão se não existirem."""
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

# --- FUNÇÕES DE PROCESSAMENTO DE IMAGEM ---

def process_document(image: Image.Image):
    """
    Detecta as bordas de um documento, desenha o contorno e retorna a imagem com perspectiva corrigida.
    """
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
            st.warning("Não foi possível detectar um documento. A imagem original será usada.", icon="⚠️")
            return image
    except Exception as e:
        st.error(f"Ocorreu um erro no processamento da imagem: {e}", icon="🚨")
        return None

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
        st.error("Erro no OCR. Verifique se o Tesseract está instalado e no PATH do sistema.", icon="🚨")
        return ""

def generate_searchable_pdf(image_list: list):
    """
    Gera um PDF pesquisável multi-páginas a partir de uma lista de imagens.
    Se o OCR falhar para uma página, ela é adicionada como uma imagem simples.
    """
    pdf_writer = PdfWriter()

    for i, image in enumerate(image_list):
        try:
            # Tenta criar um PDF de uma página com OCR
            pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(image.convert("RGB"), extension='pdf', lang='por+eng')
            pdf_page_reader = PdfReader(BytesIO(pdf_page_bytes))
            pdf_writer.add_page(pdf_page_reader.pages[0])
        except Exception as e:
            st.warning(f"OCR falhou para a página {i+1}. Adicionando como imagem. Erro: {e}", icon="⚠️")
            # --- BLOCO CORRIGIDO ---
            # Se o OCR falhar, converte a imagem para um PDF de página única
            # e o adiciona ao documento final.
            pdf_buffer = BytesIO()
            # Converte a imagem PIL para um PDF em memória
            image.convert("RGB").save(pdf_buffer, format='PDF', resolution=100.0)
            pdf_buffer.seek(0)

            # Agora lê este buffer que contém um PDF válido
            img_page_reader = PdfReader(pdf_buffer)
            pdf_writer.add_page(img_page_reader.pages[0])

    # Salva o PDF final em um buffer
    final_pdf_buffer = BytesIO()
    pdf_writer.write(final_pdf_buffer)
    final_pdf_buffer.seek(0)
    return final_pdf_buffer

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("📄 Digitalizador PRO V3")

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Finalizar Documento")
    if not st.session_state.document_pages:
        st.info("Adicione pelo menos uma página para criar um documento.")
    else:
        st.subheader(f"Páginas: {len(st.session_state.document_pages)}")

        default_filename = f"documento_digitalizado_{len(st.session_state.document_pages)}pags.pdf"
        file_name_input = st.text_input("Nome do arquivo PDF:", default_filename)

        if file_name_input and not file_name_input.lower().endswith('.pdf'):
            file_name_input += '.pdf'

        # O botão de gerar PDF agora fica aqui
        if st.button("Gerar PDF Pesquisável", use_container_width=True, type="primary"):
            with st.spinner("Gerando PDF... Isso pode levar um momento."):
                pdf_buffer = generate_searchable_pdf(st.session_state.document_pages)

                st.download_button(
                    label="📥 Baixar PDF Pronto",
                    data=pdf_buffer,
                    file_name=file_name_input,
                    mime="application/pdf",
                    use_container_width=True
                )

        if st.button("🗑️ Limpar Documento", use_container_width=True):
            st.session_state.document_pages = []
            st.toast("Documento limpo com sucesso!", icon="✨")
            st.rerun()

    st.divider()
    with st.expander("💡 Como obter os melhores resultados?"):
        st.markdown("""
        - **Fundo Contrastante:** Coloque o papel sobre uma mesa escura.
        - **Boa Iluminação:** Evite sombras sobre o documento.
        - **Enquadramento:** Capture o documento com uma pequena borda do fundo.
        - **Salvar Arquivos:** Para que o navegador pergunte onde salvar, ative a opção nas configurações de Download do seu navegador.
        """)

# --- ABAS PRINCIPAIS ---
tab1, tab2, tab3 = st.tabs([
    "1. Digitalizar e Adicionar Página",
    "2. Montar Documento",
    "3. Extrair Texto (OCR)"
])

# --- ABA 1: DIGITALIZAR E PROCESSAR ---
with tab1:
    st.header("Passo 1: Capturar Imagem")
    col1, col2 = st.columns(2)
    image_source = None
    with col1:
        uploaded_file = st.file_uploader("Upload de arquivo", type=["png", "jpg", "jpeg"], key="uploader")
    with col2:
        camera_photo = st.camera_input("Usar a câmera", key="camera")

    if camera_photo: image_source = camera_photo
    elif uploaded_file: image_source = uploaded_file

    if image_source and not st.session_state.current_image_to_process:
        try:
            st.session_state.current_image_to_process = Image.open(image_source)
            with st.spinner("Analisando imagem e detectando bordas..."):
                st.session_state.processed_image = process_document(st.session_state.current_image_to_process)
                st.session_state.final_image_to_add = st.session_state.processed_image
        except Exception as e:
            st.error(f"Não foi possível abrir o arquivo de imagem. Pode estar corrompido. Erro: {e}", icon="🚨")
            st.session_state.current_image_to_process = None

    if st.session_state.processed_image:
        st.header("Passo 2: Verificar e Otimizar")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Detecção de Bordas")
            if st.session_state.detected_contour_image:
                st.image(st.session_state.detected_contour_image, caption="Contorno detectado.", use_column_width=True)

        with col_b:
            st.subheader("Resultado para Adicionar")

            optimization_mode = st.selectbox("Modo de Otimização:", ('Cor Original', 'Otimizado para Leitura (P&B)'), key="optimization")
            rotated_image = st.session_state.processed_image

            # Controles de Rotação
            r_col1, r_col2 = st.columns(2)
            if r_col1.button("🔄 Rotacionar 90°", use_container_width=True):
                st.session_state.processed_image = st.session_state.processed_image.rotate(90, expand=True)
            if r_col2.button("Rotacionar -90° 🔄", use_container_width=True):
                 st.session_state.processed_image = st.session_state.processed_image.rotate(-90, expand=True)

            final_image = apply_post_processing(st.session_state.processed_image, optimization_mode)
            st.session_state.final_image_to_add = final_image
            st.image(final_image, caption="Imagem final ajustada.", use_column_width=True)

        st.divider()
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("✅ Adicionar ao Documento", use_container_width=True, type="primary"):
            st.session_state.document_pages.append(st.session_state.final_image_to_add)
            st.toast(f"Página {len(st.session_state.document_pages)} adicionada!", icon="🎉")
            st.session_state.current_image_to_process = None # Reseta para permitir nova imagem
            st.rerun()

        if b_col2.button("❌ Descartar Imagem", use_container_width=True):
            st.session_state.current_image_to_process = None # Reseta tudo
            st.rerun()

# --- ABA 2: MONTAR DOCUMENTO ---
with tab2:
    st.header("Ordem das Páginas do Documento")
    if not st.session_state.document_pages:
        st.info("Nenhuma página foi adicionada. Volte para a primeira aba para digitalizar.")
    else:
        for i, page_img in enumerate(st.session_state.document_pages):
            st.subheader(f"Página {i+1}")
            st.image(page_img, use_column_width=True)

            # Botões de controle
            c1, c2, c3, c4 = st.columns([2, 2, 1, 5])
            if c1.button("⬅️ Mover para Esquerda", key=f"left_{i}", disabled=(i==0), use_container_width=True):
                st.session_state.document_pages.insert(i-1, st.session_state.document_pages.pop(i))
                st.rerun()

            if c2.button("Mover para Direita ➡️", key=f"right_{i}", disabled=(i==len(st.session_state.document_pages)-1), use_container_width=True):
                st.session_state.document_pages.insert(i+1, st.session_state.document_pages.pop(i))
                st.rerun()

            if c3.button("🗑️ Remover", key=f"del_{i}", use_container_width=True, type="secondary"):
                st.session_state.document_pages.pop(i)
                st.rerun()
            st.divider()

# --- ABA 3: EXTRAIR TEXTO (OCR) ---
with tab3:
    st.header("Extração de Texto (OCR)")
    if not st.session_state.document_pages:
        st.info("Adicione páginas na primeira aba para poder extrair texto.")
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
                if st.button("Executar OCR nesta página", use_container_width=True):
                    with st.spinner("Lendo texto da imagem..."):
                        extracted_text = apply_ocr(selected_image)
                        st.text_area("Resultado:", value=extracted_text, height=400)
