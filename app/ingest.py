"""Lógica de ingesta de PDFs para RAG.

Pipeline lineal: cargar PDF → dividir en chunks → generar embeddings → guardar en DB.
No necesita LangGraph porque no es un flujo agentico, es un proceso secuencial.

Usa:
- pypdf para extraer texto del PDF (ya en requirements)
- langchain RecursiveCharacterTextSplitter para chunking (ya en requirements)
- all-MiniLM-L6-v2 (sentence-transformers) para embeddings (384 dims, local, gratuito)
- psycopg pool existente para guardar en Supabase (utils/database.py)
"""

import io
from typing import Optional

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from utils.database import get_connection


# --- Configuración ---

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TIPOS_VALIDOS = ["politica_garantia", "catalogo", "procedimiento", "faq", "otro"]

# Modelo de embeddings: se carga una sola vez al importar el módulo
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def generar_embeddings(chunks: list[dict]) -> list[list[float]]:
    """Genera embeddings para los chunks usando all-MiniLM-L6-v2.

    Procesa todos los textos de una sola vez (modelo local, sin límites de API).
    Retorna listas de floats (384 dims) compatibles con pgvector.
    """
    textos = [c["text"] for c in chunks]

    embeddings = _embed_model.encode(textos, show_progress_bar=False)
    all_embeddings = [emb.tolist() for emb in embeddings]

    print(f"INGEST - Embeddings generados: {len(all_embeddings)} ({len(all_embeddings[0])} dims)")

    return all_embeddings


async def guardar_documento(
    nombre: str,
    tipo: str,
    descripcion: str,
    filename: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> str:
    """Guarda el documento y sus chunks en la base de datos.

    Usa el pool de conexiones existente (psycopg).
    Retorna el id_documento generado.
    """
    async with get_connection() as conn:
        # 1. Insertar documento principal
        cursor = await conn.execute(
            """
            INSERT INTO documents (nombre, descripcion, tipo, activo)
            VALUES (%s, %s, %s, true)
            RETURNING id::text
            """,
            (nombre, descripcion, tipo),
        )
        row = await cursor.fetchone()
        id_documento = row[0]

        print(f"INGEST - Documento creado: {id_documento}")

        # 2. Insertar chunks con embeddings
        for chunk, embedding in zip(chunks, embeddings):
            await conn.execute(
                """
                INSERT INTO document_chunks
                    (id_documento, contenido, embedding, pagina, chunk_index, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    id_documento,
                    chunk["text"],
                    embedding,
                    chunk["page"],
                    chunk["chunk_index"],
                    {"source": filename, "page": chunk["page"]},
                ),
            )

        print(f"INGEST - {len(chunks)} chunks insertados")

    return id_documento


async def ingest_pdf(
    file_bytes: bytes,
    nombre: str,
    tipo: str,
    descripcion: str = "",
    filename: str = "documento.pdf",
) -> dict:
    """Orquesta el pipeline completo de ingesta.

    1. Extraer texto del PDF
    2. Dividir en chunks
    3. Generar embeddings con all-MiniLM-L6-v2
    4. Guardar en base de datos

    Retorna dict con el resultado de la ingesta.
    """
    print(f"\nINGEST - Iniciando: {nombre} ({tipo})")

    # 1. Extraer texto
    paginas = extraer_texto_pdf(file_bytes)
    if not paginas:
        raise ValueError("El PDF no contiene texto extraíble")
    print(f"INGEST - {len(paginas)} páginas con texto")

    # 2. Dividir en chunks
    chunks = dividir_en_chunks(paginas)
    if not chunks:
        raise ValueError("No se generaron chunks válidos del PDF")
    print(f"INGEST - {len(chunks)} chunks generados")

    # 3. Generar embeddings
    embeddings = generar_embeddings(chunks)

    # 4. Guardar en DB
    id_documento = await guardar_documento(
        nombre=nombre,
        tipo=tipo,
        descripcion=descripcion,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"INGEST - Completado: {id_documento}")

    return {
        "id_documento": id_documento,
        "chunks_insertados": len(chunks),
        "paginas_procesadas": len(paginas),
        "nombre": nombre,
        "tipo": tipo,
    }
