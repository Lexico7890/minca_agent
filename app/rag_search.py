"""Nodo de búsqueda semántica (RAG) sobre documentos ingestados.

Pipeline:
1. Genera embedding de la pregunta del usuario (Gemini text-embedding-004)
2. Busca chunks similares en document_chunks usando match_documents() (pgvector HNSW)
3. Retorna los chunks más relevantes en contexto_rag

Usa el MISMO modelo de embeddings que ingest.py para garantizar
consistencia entre los vectores almacenados y los de búsqueda.
"""

import os
from app.state import AgentState
from utils.database import get_connection


# --- Configuración ---

EMBED_MODEL = "text-embedding-004"
MATCH_THRESHOLD = 0.5   # Similitud mínima (coseno). Conservador para español.
MATCH_COUNT = 5          # Top-k chunks a retornar


# --- Generación de embedding para la query ---

def generar_embedding_query(texto: str) -> list[float]:
    """Genera un embedding para la pregunta del usuario.

    Usa Gemini text-embedding-004 (mismo modelo que ingest.py).
    Un solo texto, sin batching — es una query individual.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY no configurada")

    from google import genai
    client = genai.Client(api_key=gemini_key)

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[texto],
    )

    embedding = result.embeddings[0].values
    print(f"BUSCADOR_RAG - Embedding generado ({len(embedding)} dims)")
    return embedding


# --- Nodo principal del buscador RAG ---

async def buscar_documentos(state: AgentState) -> dict:
    """Nodo de búsqueda semántica: genera embedding y busca en document_chunks.

    Usa la función match_documents() creada en la migración,
    que ejecuta búsqueda vectorial con HNSW + cosine distance.

    Retorna contexto_rag con los chunks más relevantes.
    """
    pregunta = state.pregunta_actual

    if not pregunta.strip():
        return {
            "errores": [{
                "nodo": "buscador_rag",
                "mensaje": "Pregunta vacía para búsqueda RAG",
                "recuperable": False
            }]
        }

    try:
        # 1. Generar embedding de la pregunta
        embedding = generar_embedding_query(pregunta)

        # 2. Buscar chunks similares usando match_documents()
        # La función retorna: id, contenido, pagina, chunk_index,
        #                      documento_nombre, documento_tipo, similarity
        async with get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM match_documents(%s::vector(768), %s, %s)",
                (str(embedding), MATCH_THRESHOLD, MATCH_COUNT),
            )
            rows = await cursor.fetchall()

            if cursor.description:
                columnas = [desc[0] for desc in cursor.description]
            else:
                columnas = []

            resultados = [dict(zip(columnas, row)) for row in rows]

        if not resultados:
            print("BUSCADOR_RAG - Sin resultados sobre el threshold")
            return {"intenciones": ["rag_sin_resultados"]}

        # 3. Formatear para contexto_rag
        chunks = []
        for r in resultados:
            chunks.append({
                "contenido": r["contenido"],
                "pagina": r.get("pagina"),
                "chunk_index": r.get("chunk_index"),
                "documento": r.get("documento_nombre", ""),
                "tipo": r.get("documento_tipo", ""),
                "similitud": round(float(r.get("similarity", 0)), 3),
            })

        max_sim = max(c["similitud"] for c in chunks)
        print(f"BUSCADOR_RAG - {len(chunks)} chunks encontrados (similitud max: {max_sim:.2f})")

        return {
            "contexto_rag": chunks,
            "intenciones": ["rag_documentos"],
        }

    except ValueError as e:
        # Error de configuración (GEMINI_API_KEY)
        print(f"BUSCADOR_RAG - Error config: {str(e)}")
        return {
            "errores": [{
                "nodo": "buscador_rag",
                "mensaje": str(e),
                "recuperable": False
            }]
        }

    except Exception as e:
        error_msg = str(e)[:300]
        print(f"BUSCADOR_RAG - Error: {error_msg}")
        return {
            "errores": [{
                "nodo": "buscador_rag",
                "mensaje": f"Error en búsqueda RAG: {error_msg}",
                "recuperable": False
            }]
        }
