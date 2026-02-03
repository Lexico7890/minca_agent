"""Aplicación FastAPI del servicio de agentes.

Este es el servidor HTTP que las Edge Functions de Supabase
llaman para procesar preguntas del usuario.

MANEJO DE SESIONES:
El frontend envía un session_id con cada petición. Este ID
representa una conversación. Usamos un diccionario en memoria
para guardar la memoria de cada sesión entre peticiones.

En producción esto podría moverse a Redis o una tabla de DB,
pero para el MVP un diccionario es suficiente y más simple.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
import os
import uuid

from utils.database import startup, shutdown
from app.graph import agent
from app.state import AgentState, MensajeMemoria


# --- Almacén de sesiones en memoria ---
# Clave: session_id | Valor: lista de MensajeMemoria
_sesiones: Dict[str, List[MensajeMemoria]] = {}

# Límite de sesiones activas para evitar que la memoria crezca infinita
MAX_SESIONES = 100


# --- Modelos de petición y respuesta ---

class PreguntaRequest(BaseModel):
    """Cuerpo de la petición desde la Edge Function."""
    pregunta: str
    session_id: Optional[str] = None
    """ID de sesión opcional. Si no viene, se crea una nueva sesión.
    El frontend debe guardar este ID y enviarlo en peticiones siguientes
    para mantener el contexto de la conversación."""


class RespuestaResponse(BaseModel):
    """Cuerpo de la respuesta que retorna al Edge Function."""
    respuesta: str
    session_id: str
    intenciones_detectadas: List[str]
    errores: Optional[List[dict]] = None


# --- Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicio y cierre del servicio."""
    await startup()  # Inicializa el pool de DB
    yield
    await shutdown()  # Cierra el pool


# --- Aplicación ---

app = FastAPI(
    title="Minca Electric - Servicio de Agentes",
    lifespan=lifespan
)


# --- Autenticación interna ---

async def verificar_autenticacion(request: Request):
    """Verifica el secret compartido con las Edge Functions.
    
    Este endpoint nunca debe ser accesible desde el frontend directamente.
    El secret asegura que solo las Edge Functions de Supabase pueden llamarlo.
    """
    secret = os.getenv("AGENT_SERVICE_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="AGENT_SERVICE_SECRET no configurado")

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="No autorizado")


# --- Gestión de sesiones ---

def obtener_memoria_sesion(session_id: Optional[str]) -> tuple[str, List[MensajeMemoria]]:
    """Obtiene o crea una sesión.
    
    Retorna una tupla con (session_id, memoria_actual).
    Si el session_id no existe o es None, crea una nueva sesión.
    """
    if session_id and session_id in _sesiones:
        return session_id, _sesiones[session_id]

    # Crear nueva sesión
    nuevo_id = str(uuid.uuid4())
    _sesiones[nuevo_id] = []

    # Si hay demasiadas sesiones, eliminar las más antiguas
    if len(_sesiones) > MAX_SESIONES:
        claves = list(_sesiones.keys())
        # Eliminar las primeras (más antiguas)
        for clave in claves[:len(_sesiones) - MAX_SESIONES]:
            del _sesiones[clave]

    return nuevo_id, _sesiones[nuevo_id]


def guardar_memoria_sesion(session_id: str, memoria: List[MensajeMemoria]):
    """Actualiza la memoria de una sesión después de procesar una pregunta."""
    _sesiones[session_id] = memoria


# --- Endpoint principal ---

@app.post("/procesar-pregunta", response_model=RespuestaResponse)
async def procesar_pregunta(request: Request, body: PreguntaRequest):
    """Procesa una pregunta del usuario y retorna la respuesta del agente.
    
    Flujo:
    1. Verificar autenticación
    2. Validar entrada
    3. Obtener memoria de la sesión
    4. Crear estado inicial con la pregunta y la memoria
    5. Ejecutar el grafo de LangGraph
    6. Guardar la memoria actualizada
    7. Retornar la respuesta
    """

    # 1. Autenticación
    await verificar_autenticacion(request)

    # 2. Validar que la pregunta no esté vacía
    pregunta = body.pregunta.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    # 3. Obtener sesión
    session_id, memoria_actual = obtener_memoria_sesion(body.session_id)

    # 4. Crear estado inicial como diccionario.
    # LangGraph ainvoke() espera un dict como entrada y retorna un dict.
    # Solo pasamos los campos que cambian; el resto usa valores por defecto
    # de AgentState automáticamente.
    estado_inicial = {
        "pregunta_actual": pregunta,
        "memoria": [msg.model_dump() for msg in memoria_actual]
    }

    # 5. Ejecutar el grafo
    # ainvoke() ejecuta todos los nodos según la estructura del grafo
    # y retorna el estado final como diccionario.
    resultado = await agent.ainvoke(estado_inicial)
    
    # IMPORTANTE: LangGraph solo retorna los campos que cambiaron durante
    # la ejecución. Los campos con valores por defecto que nunca se modifican
    # no aparecen en el resultado. Usamos .get() con defaults.

    # 6. Guardar memoria actualizada
    # El generador de respuesta ya actualizó memoria en el estado.
    # resultado["memoria"] puede venir como lista de dicts o lista de objetos MensajeMemoria
    # dependiendo de cómo LangGraph lo maneje internamente.
    memoria_actualizada = []
    for msg in resultado.get("memoria", []):
        if isinstance(msg, MensajeMemoria):
            # Ya es un objeto MensajeMemoria, usarlo directamente
            memoria_actualizada.append(msg)
        elif isinstance(msg, dict):
            # Es un dict, convertirlo a MensajeMemoria
            memoria_actualizada.append(MensajeMemoria(**msg))
    
    guardar_memoria_sesion(session_id, memoria_actualizada)

    # 7. Retornar respuesta
    return RespuestaResponse(
        respuesta=resultado.get("respuesta_final", "No se pudo generar una respuesta"),
        session_id=session_id,
        intenciones_detectadas=resultado.get("intenciones", []),
        errores=resultado.get("errores", None)
    )


# --- Endpoint de salud ---

@app.get("/health")
async def health_check():
    """Verificar que el servicio está vivo. Usado por Railway/Render."""
    return {"status": "ok"}