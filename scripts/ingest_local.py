"""Script local para ingesta de PDFs en la base de datos.

Corre en tu máquina local (no en Render) porque necesita RAM para cargar
el modelo de embeddings all-MiniLM-L6-v2.

Uso:
    python scripts/ingest_local.py ruta/al/archivo.pdf \
        --nombre "Política de Garantía" \
        --tipo politica_garantia \
        --descripcion "Versión 2024"

Tipos válidos: politica_garantia, catalogo, procedimiento, faq, otro

Requisitos:
    - .env con SUPABASE_DB_URL definida
    - pip install sentence-transformers psycopg python-dotenv pypdf langchain
"""

import sys
import os
import json
import argparse

# Agregar el directorio raíz del proyecto al path para poder importar app/ingest.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer

from app.ingest import extraer_texto_pdf, dividir_en_chunks, TIPOS_VALIDOS


def generar_embeddings(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Genera embeddings para los chunks usando all-MiniLM-L6-v2.

    Procesa todos los textos de una sola vez (modelo local, sin límites de API).
    Retorna listas de floats (384 dims) compatibles con pgvector.
    """
    textos = [c["text"] for c in chunks]
    embeddings = model.encode(textos, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]


def guardar_documento(
    conn: psycopg.Connection,
    nombre: str,
    tipo: str,
    descripcion: str,
    filename: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> str:
    """Guarda el documento y sus chunks en la base de datos.

    Usa una conexión psycopg síncrona.
    Retorna el id_documento generado.
    """
    with conn.transaction():
        # 1. Insertar documento principal
        cursor = conn.execute(
            """
            INSERT INTO documents (nombre, descripcion, tipo, activo)
            VALUES (%s, %s, %s, true)
            RETURNING id::text
            """,
            (nombre, descripcion, tipo),
        )
        row = cursor.fetchone()
        id_documento = row[0]

        print(f"  Documento creado: {id_documento}")

        # 2. Insertar chunks con embeddings
        for chunk, embedding in zip(chunks, embeddings):
            conn.execute(
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
                    json.dumps({"source": filename, "page": chunk["page"]}),
                ),
            )

        print(f"  {len(chunks)} chunks insertados")

    return id_documento


def main():
    parser = argparse.ArgumentParser(
        description="Ingesta local de PDFs para RAG (corre en tu máquina, no en Render)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Tipos válidos: {', '.join(TIPOS_VALIDOS)}",
    )
    parser.add_argument("pdf", help="Ruta al archivo PDF")
    parser.add_argument("--nombre", required=True, help="Nombre descriptivo del documento")
    parser.add_argument("--tipo", required=True, choices=TIPOS_VALIDOS, help="Tipo de documento")
    parser.add_argument("--descripcion", default="", help="Descripción opcional del documento")

    args = parser.parse_args()

    # 1. Validar archivo
    if not os.path.isfile(args.pdf):
        print(f"Error: No se encontró el archivo: {args.pdf}")
        sys.exit(1)

    if not args.pdf.lower().endswith(".pdf"):
        print("Error: Solo se aceptan archivos PDF")
        sys.exit(1)

    # 2. Cargar variables de entorno
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("Error: SUPABASE_DB_URL no está definida en .env")
        sys.exit(1)

    # 3. Leer PDF
    print(f"\n{'='*60}")
    print(f"INGESTA LOCAL: {args.nombre} ({args.tipo})")
    print(f"{'='*60}")

    with open(args.pdf, "rb") as f:
        file_bytes = f.read()

    print(f"\n[1/5] Extrayendo texto del PDF...")
    paginas = extraer_texto_pdf(file_bytes)
    if not paginas:
        print("Error: El PDF no contiene texto extraíble")
        sys.exit(1)
    print(f"  {len(paginas)} páginas con texto")

    # 4. Dividir en chunks
    print(f"[2/5] Dividiendo en chunks...")
    chunks = dividir_en_chunks(paginas)
    if not chunks:
        print("Error: No se generaron chunks válidos del PDF")
        sys.exit(1)
    print(f"  {len(chunks)} chunks generados")

    # 5. Generar embeddings
    print(f"[3/5] Cargando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[4/5] Generando embeddings ({len(chunks)} chunks)...")
    embeddings = generar_embeddings(chunks, model)
    print(f"  Embeddings generados ({len(embeddings[0])} dims)")

    # 6. Guardar en base de datos
    print(f"[5/5] Guardando en base de datos...")
    with psycopg.connect(db_url) as conn:
        id_documento = guardar_documento(
            conn=conn,
            nombre=args.nombre,
            tipo=args.tipo,
            descripcion=args.descripcion,
            filename=os.path.basename(args.pdf),
            chunks=chunks,
            embeddings=embeddings,
        )

    print(f"\n{'='*60}")
    print(f"COMPLETADO")
    print(f"  Documento: {args.nombre}")
    print(f"  Tipo: {args.tipo}")
    print(f"  ID: {id_documento}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Páginas: {len(paginas)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
