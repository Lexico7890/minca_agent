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


async def ejecutar_consultas(state: AgentState):
    """Ejecuta las consultas según las intenciones detectadas.
    
    IMPORTANTE: Este nodo retorna un dict con los campos actualizados,
    no el estado completo. Así LangGraph puede mergear los cambios correctamente.
    """

    # Si hay error fatal, no seguimos
    if tiene_error_fatal(state):
        return {}

    # Si la pregunta no fue reconocida, no hay nada que consultar
    if state.intenciones == ["no_reconocida"]:
        return {}

    # Filtrar solo las intenciones que tienen herramienta disponible
    intenciones_validas = [
        i for i in state.intenciones
        if i in HERRAMIENTAS
    ]
    
    # DEBUG
    print(f"EJECUTOR - Intenciones a ejecutar: {intenciones_validas}")

    # Si ninguna intención tiene herramienta
    if not intenciones_validas:
        return {
            "errores": state.errores + [{
                "nodo": "ejecutor_consultas",
                "mensaje": f"No hay herramientas para las intenciones detectadas: {state.intenciones}",
                "recuperable": True
            }]
        }

    # Ejecutar las consultas y acumular resultados
    contexto_acumulado = []
    errores_acumulados = list(state.errores)  # Copiar errores existentes
    
    for intencion in intenciones_validas:
        herramienta = HERRAMIENTAS[intencion]
        # Cada herramienta recibe el estado ORIGINAL y retorna un estado NUEVO
        resultado = await herramienta(state)
        
        # Agregar los bloques de contexto que retornó esta herramienta
        contexto_acumulado.extend(resultado.contexto_db)
        
        # Agregar errores nuevos si los hay
        if resultado.errores:
            errores_acumulados.extend(resultado.errores)
    
    # DEBUG
    print(f"EJECUTOR - Contexto acumulado: {len(contexto_acumulado)} bloques")
    for bloque in contexto_acumulado:
        print(f"  EJECUTOR - {bloque['fuente']}: {len(bloque['datos'])} registros")
    
    # Retornar SOLO los campos que cambiaron
    return {
        "contexto_db": contexto_acumulado,
        "errores": errores_acumulados
    }