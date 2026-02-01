"""Estado del grafo de LangGraph.

Este es el objeto que viaja por todos los nodos del grafo.
Cada nodo lo recibe, lo puede modificar, y lo pasa al siguiente.
Es el único mecanismo de comunicación entre nodos.

DISEÑO IMPORTANTE:
- "memoria" almacena el historial de la conversación actual.
  Esto permite que el agente tenga contexto de preguntas anteriores
  dentro de la misma sesión.
- "errores" es una lista, no un solo string. Esto permite que
  si un nodo falla, el grafo pueda seguir con otros nodos
  y reportar todos los errores al final.
- "reintentos" permite que el grafo reintente un nodo
  automáticamente si falla por un error transitorio.
- La estructura ya tiene campos para operaciones de escritura
  que implementaremos después (tipo_operacion, confirmacion_usuario).
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# --- Tipos de operación que soportará el agente en el futuro ---

class TipoOperacion(str, Enum):
    """Tipos de operaciones que puede realizar el agente.
    
    Por ahora solo implementamos LECTURA.
    Las demás se agregan cuando estemos listos para escritura.
    """
    LECTURA = "lectura"
    INSERTAR = "insertar"       # Futuro
    ACTUALIZAR = "actualizar"   # Futuro
    ELIMINAR = "eliminar"       # Futuro


# --- Estructura de un mensaje de memoria ---

class MensajeMemoria(BaseModel):
    """Un mensaje individual en el historial de la conversación.
    
    rol: "usuario" para lo que dijo el usuario, "agente" para la respuesta.
    contenido: el texto del mensaje.
    """
    rol: str       # "usuario" o "agente"
    contenido: str


# --- Estado principal del grafo ---

class AgentState(BaseModel):
    """Estado completo que fluye por todos los nodos del grafo.
    
    Se divide en secciones lógicas para que sea fácil entender
    qué parte del pipeline modifica qué campo.
    """

    # =========================================================
    # ENTRADA
    # =========================================================
    pregunta_actual: str = ""
    """La pregunta que el usuario acaba de hacer."""

    # =========================================================
    # MEMORIA DE CONVERSACIÓN
    # =========================================================
    memoria: list[MensajeMemoria] = Field(default_factory=list)
    """Historial de la conversación actual (dentro de la misma sesión).
    
    Cuando el usuario hace una nueva pregunta, antes de procesarla
    agregamos el historial previo como contexto. Esto permite que
    el agente entienda referencias como "el mismo repuesto del que
    hablamos antes" o "y cuántas garantías tiene ese".
    
    Se limita a los últimos N mensajes para no exceder el contexto del modelo.
    """

    # =========================================================
    # CLASIFICACIÓN
    # =========================================================
    intenciones: list[str] = Field(default_factory=list)
    """Intenciones detectadas por el clasificador.
    
    Ejemplos: ["inventario"], ["garantias", "inventario"], ["no_reconocida"]
    """

    tipo_operacion: TipoOperacion = TipoOperacion.LECTURA
    """Tipo de operación que detectó el clasificador.
    
    Por ahora siempre será LECTURA. Cuando implementemos escritura,
    el clasificador también detectará si el usuario quiere insertar,
    actualizar o eliminar.
    """

    # =========================================================
    # CONTEXTO DE CONSULTAS
    # =========================================================
    contexto_db: list[dict] = Field(default_factory=list)
    """Resultados de las consultas a PostgreSQL.
    
    Cada elemento tiene:
    - "fuente": nombre de la categoría consultada
    - "datos": lista de filas retornadas como diccionarios
    
    Ejemplo:
    [
        {
            "fuente": "inventario",
            "datos": [
                {"repuesto": "Filtro X", "cantidad": 15, "localizacion": "Bodega A"},
                {"repuesto": "Filtro Y", "cantidad": 3, "localizacion": "Bodega B"}
            ]
        }
    ]
    """

    contexto_rag: list[dict] = Field(default_factory=list)
    """Resultados de búsqueda semántica contra PDFs.
    
    Se implementará en la siguiente fase. El campo existe ahora
    para que la estructura del estado no cambie cuando lo agregues.
    """

    # =========================================================
    # OPERACIONES DE ESCRITURA (futuro)
    # =========================================================
    confirmacion_usuario: bool = False
    """Si el usuario confirmó una operación de escritura.
    
    Cuando el agente detecte que el usuario quiere modificar datos,
    primero debe pedir confirmación antes de ejecutar. Este campo
    tracking ese estado de confirmación.
    """

    # =========================================================
    # CONTROL DE FLUJO Y ERRORES
    # =========================================================
    errores: list[dict] = Field(default_factory=list)
    """Lista de errores que ocurrieron durante el procesamiento.
    
    Cada error tiene:
    - "nodo": en qué nodo del grafo ocurrió
    - "mensaje": descripción del error
    - "recuperable": si el grafo puede seguir sin ese resultado
    
    Ejemplo:
    [{"nodo": "consulta_inventario", "mensaje": "timeout", "recuperable": True}]
    
    Un error recuperable significa que esa consulta falló pero las demás
    siguieron adelante. El modelo puede generar una respuesta parcial.
    Un error no recuperable detiene todo.
    """

    reintentos_restantes: int = 2
    """Cantidad de reintentos disponibles para el nodo actual.
    
    Si un nodo falla con un error recuperable (como timeout),
    el grafo puede intentarlo de nuevo hasta que llegue a 0.
    Se resetea para cada nodo.
    """

    # =========================================================
    # RESPUESTA FINAL
    # =========================================================
    respuesta_final: str = ""
    """Texto humanizado que va a convertirse en voz para el usuario."""