"""Estado del grafo de LangGraph - VERSIÓN CORREGIDA.

IMPORTANTE: LangGraph con Pydantic BaseModel tiene problemas cuando
mut as listas/dicts directamente (con .append()). La solución es usar
Annotated con operator.add, que le dice a LangGraph cómo "mergear"
los cambios entre nodos.

Con esto, cuando un nodo retorna {"contexto_db": [nuevo_item]},
LangGraph automáticamente hace contexto_db += [nuevo_item] en lugar
de reemplazar la lista completa.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Annotated
from enum import Enum
import operator


# --- Tipos de operación que soportará el agente en el futuro ---

class TipoOperacion(str, Enum):
    """Tipo de operación que el usuario quiere hacer."""
    LECTURA = "lectura"
    INSERTAR = "insertar"
    ACTUALIZAR = "actualizar"
    ELIMINAR = "eliminar"


class MensajeMemoria(BaseModel):
    """Un mensaje individual en el historial de la conversación."""
    rol: str       # "usuario" o "agente"
    contenido: str


# --- Estado principal del grafo ---

class AgentState(BaseModel):
    """Estado completo que fluye por todos los nodos del grafo.
    
    IMPORTANTE: Los campos con Annotated[List[X], operator.add] se
    acumulan entre nodos. Si un nodo retorna {"errores": [nuevo_error]},
    LangGraph hace state.errores += [nuevo_error] automáticamente.
    """

    # === ENTRADA ===
    pregunta_actual: str = ""
    memoria: Annotated[List[MensajeMemoria], operator.add] = Field(default_factory=list)
    
    # === CLASIFICACIÓN ===
    intenciones: List[str] = Field(default_factory=list)
    tipo_operacion: TipoOperacion = TipoOperacion.LECTURA
    
    # === CONTEXTO DE DATOS ===
    contexto_db: Annotated[List[Dict], operator.add] = Field(default_factory=list)
    contexto_rag: Annotated[List[Dict], operator.add] = Field(default_factory=list)
    
    # === MANEJO DE ERRORES ===
    errores: Annotated[List[Dict], operator.add] = Field(default_factory=list)
    reintentos_restantes: int = 3
    
    # === FUTURO: ESCRITURA ===
    confirmacion_usuario: bool = False
    
    # === SALIDA ===
    respuesta_final: str = ""