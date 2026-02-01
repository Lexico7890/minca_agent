"""Nodo ejecutor de consultas.

Este nodo lee las intenciones del clasificador y ejecuta
las consultas correspondientes en paralelo.

LÓGICA DE CONTROL:
1. Primero verifica si hay algún error no recuperable.
   Si existe, no tiene sentido consultar nada.
2. Filtra las intenciones que tienen herramienta disponible.
3. Si hay varias intenciones, las ejecuta en paralelo con asyncio.
4. Cada consulta maneja sus propios errores internamente,
   así que si una falla las demás siguen adelante.
"""

import asyncio
from app.state import AgentState
from tools.db_tools import HERRAMIENTAS


def tiene_error_fatal(state: AgentState) -> bool:
    """Verifica si hay algún error no recuperable en el estado.
    
    Un error no recuperable significa que algo fundamental falló
    (como la clasificación) y no tiene sentido seguir procesando.
    """
    return any(not e["recuperable"] for e in state.errores)


async def ejecutar_consultas(state: AgentState) -> AgentState:
    """Ejecuta las consultas según las intenciones detectadas.
    
    Este nodo es donde LangGraph nos da valor real:
    podemos despachar múltiples consultas en paralelo y
    combinar los resultados en un solo estado, sin tener
    que manejar esa lógica manualmente fuera del grafo.
    """

    # Si hay error fatal, no seguimos
    if tiene_error_fatal(state):
        return state

    # Si la pregunta no fue reconocida, no hay nada que consultar
    if state.intenciones == ["no_reconocida"]:
        return state

    # Filtrar solo las intenciones que tienen herramienta disponible
    intenciones_validas = [
        i for i in state.intenciones
        if i in HERRAMIENTAS
    ]

    # Si ninguna intención tiene herramienta, registrar que no se encontró
    if not intenciones_validas:
        state.errores.append({
            "nodo": "ejecutor_consultas",
            "mensaje": f"No hay herramientas para las intenciones detectadas: {state.intenciones}",
            "recuperable": True
        })
        return state

    # Crear las tareas de consulta
    tareas = [HERRAMIENTAS[intencion](state) for intencion in intenciones_validas]

    # Ejecutar en paralelo si hay más de una
    if len(tareas) == 1:
        await tareas[0]
    else:
        await asyncio.gather(*tareas)

    # Verificar que al menos una consulta retornó datos
    if not state.contexto_db:
        state.errores.append({
            "nodo": "ejecutor_consultas",
            "mensaje": "Ninguna consulta retornó datos",
            "recuperable": True  # El modelo puede responder diciendo que no hay datos
        })

    return state