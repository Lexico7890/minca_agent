"""Definición del grafo de LangGraph con SQL dinámico.

ESTRUCTURA DEL GRAFO:

    START
      ↓
    generador_sql
      ↓
    ┌── decide_sql ──┐
    ↓                ↓
  skip_sql       ejecutor_sql
    ↓                ↓
    │          ┌── decide_exec ──┐
    │          ↓                 ↓
    │     retry → generador_sql  normal
    │                            ↓
    └────────────┬───────────────┘
                 ↓
         generador_respuesta
                 ↓
                END

El loop de reintento permite que el LLM corrija su SQL
hasta 2 veces si la ejecución falla.
"""

from langgraph.graph import StateGraph, START, END
from app.state import AgentState
from app.sql_generator import generar_sql
from app.sql_executor import ejecutar_sql
from app.response_generator import generar_respuesta


def _tiene_error_fatal(state: AgentState) -> bool:
    """Verifica si hay algún error no recuperable."""
    return any(not e.get("recuperable", True) for e in state.errores)


def decidir_despues_de_sql(state: AgentState) -> str:
    """Decisión después del generador de SQL.

    Rutas posibles:
    - Error fatal (API falló) → generador_respuesta
    - Saludo / no reconocida → generador_respuesta
    - SQL generado → ejecutor_sql
    """
    if _tiene_error_fatal(state):
        return "generador_respuesta"

    if "saludo" in state.intenciones or state.intenciones == ["no_reconocida"]:
        return "generador_respuesta"

    if state.sql_generado:
        return "ejecutor_sql"

    return "generador_respuesta"


def decidir_despues_de_ejecucion(state: AgentState) -> str:
    """Decisión después del ejecutor de SQL.

    Rutas posibles:
    - Datos obtenidos → generador_respuesta
    - Error + reintentos disponibles → generador_sql (retry)
    - Error + sin reintentos → generador_respuesta
    """
    # Si tenemos datos, seguimos al generador de respuesta
    if state.contexto_db:
        return "generador_respuesta"

    # Si falló y aún hay reintentos, volver al generador de SQL
    if state.sql_error_anterior and state.sql_reintentos > 0:
        return "generador_sql"

    # Sin datos y sin reintentos, ir al generador (manejará resultado vacío)
    return "generador_respuesta"


def construir_grafo():
    """Construye y compila el grafo del agente con SQL dinámico."""
    grafo = StateGraph(AgentState)

    # --- Registrar nodos ---
    grafo.add_node("generador_sql", generar_sql)
    grafo.add_node("ejecutor_sql", ejecutar_sql)
    grafo.add_node("generador_respuesta", generar_respuesta)

    # --- Arista de entrada ---
    grafo.add_edge(START, "generador_sql")

    # --- Arista condicional después del generador de SQL ---
    grafo.add_conditional_edges(
        "generador_sql",
        decidir_despues_de_sql
    )

    # --- Arista condicional después del ejecutor (permite retry) ---
    grafo.add_conditional_edges(
        "ejecutor_sql",
        decidir_despues_de_ejecucion
    )

    # --- Arista final ---
    grafo.add_edge("generador_respuesta", END)

    return grafo.compile()


# Instancia única del grafo compilado.
agent = construir_grafo()
