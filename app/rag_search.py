"""Nodo de búsqueda semántica (RAG) sobre documentos ingestados.

Pipeline:
1. Genera embedding de la pregunta del usuario (all-MiniLM-L6-v2, 384 dims)
2. Busca chunks similares en document_chunks usando match_documents() (pgvector HNSW)
3. Retorna los chunks más relevantes en contexto_rag

Usa el MISMO modelo de embeddings que scripts/ingest_local.py para garantizar
consistencia entre los vectores almacenados y los de búsqueda.

NOTA: El modelo se carga con lazy loading (al primer uso, no al importar)
para reducir el consumo de RAM en Render free tier.
"""

from app.state import AgentState
from utils.database import get_connection


# --- Configuración ---

MATCH_THRESHOLD = 0.5   # Similitud mínima (coseno). Conservador para español.
MATCH_COUNT = 5          # Top-k chunks a retornar

# Modelo de embeddings: lazy loading para no consumir RAM al arrancar
_embed_model = None


def _get_embed_model():
    """Carga el modelo de embeddings la primera vez que se necesita.

    Lazy loading: evita cargar ~80MB de RAM al arrancar el servidor.
    Solo se carga cuando llega la primera query RAG.
    """
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("BUSCADOR_RAG - Cargando modelo de embeddings (primera vez)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("BUSCADOR_RAG - Modelo cargado")
    return _embed_model


# --- Generación de embedding para la query ---

def generar_embedding_query(texto: str) -> list[float]:
    """Genera un embedding para la pregunta del usuario.

    Usa all-MiniLM-L6-v2 (mismo modelo que scripts/ingest_local.py).
    Un solo texto, sin batching — es una query individual.
    """
    model = _get_embed_model()
    embedding = model.encode([texto])[0].tolist()
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
                "SELECT * FROM match_documents(%s::vector(384), %s, %s)",
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
