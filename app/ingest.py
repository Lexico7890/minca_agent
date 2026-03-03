"""Utilidades de procesamiento de texto para ingesta de PDFs.

Funciones puras de extracción y chunking reutilizables por
el script local de ingesta (scripts/ingest_local.py).

NO carga modelos de embeddings ni accede a la base de datos.
Eso lo hace el script local directamente.
"""

import io

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Configuración ---

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TIPOS_VALIDOS = ["politica_garantia", "catalogo", "procedimiento", "faq", "manual_usuario", "otro"]


# --- Funciones de procesamiento ---

def extraer_texto_pdf(file_bytes: bytes) -> list[dict]:
    """Extrae texto de cada página del PDF.

    Retorna lista de dicts: [{"page": 1, "text": "contenido..."}, ...]
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    paginas = []

    for i, page in enumerate(reader.pages):
        texto = page.extract_text() or ""
        texto = texto.strip()
        if texto:
            paginas.append({"page": i + 1, "text": texto})

    return paginas


def dividir_en_chunks(paginas: list[dict]) -> list[dict]:
    """Divide el texto de las páginas en chunks con metadata.

    Retorna lista de dicts: [{"text": "chunk...", "page": 1, "chunk_index": 0}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0

    for pagina in paginas:
        textos_pagina = splitter.split_text(pagina["text"])
        for texto in textos_pagina:
            texto_limpio = texto.strip()
            if len(texto_limpio) > 50:  # Filtrar chunks muy cortos
                chunks.append({
                    "text": texto_limpio,
                    "page": pagina["page"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

    return chunks
