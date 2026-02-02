"""Definición del grafo de LangGraph.

Aquí se conectan todos los nodos y se definen las aristas.
Este archivo es donde LangGraph nos da valor real:
los nodos son funciones normales de Python, pero la lógica
de flujo (qué se ejecuta después de qué, y bajo qué condiciones)
la define el grafo, no nosotros con ifs anidados.

ESTRUCTURA DEL GRAFO:

    clasificador
         ↓
    ┌─── decide ───┐
    ↓              ↓
  error_fatal    ejecutor_consultas
    ↓              ↓
    └──────┬───────┘
           ↓
    generador_respuesta
           ↓
         END

La arista condicional después del clasificador es donde
LangGraph muestra su valor: si hay un error fatal (por ejemplo
Gemini no respondió), saltamos directamente al generador
sin intentar consultar la base de datos.
"""

from langgraph.graph import StateGraph, START, END
from app.state import AgentState
from app.classifier import clasificar_pregunta
from app.query_executor import ejecutar_consultas, tiene_error_fatal
from app.response_generator import generar_respuesta


def decidir_después_de_clasificador(state: AgentState) -> str:
    """Función de decisión después del clasificador.
    
    Este es el patrón clave de LangGraph: las aristas condicionales.
    En lugar de tener un if dentro del ejecutor que verifica si puede
    seguir, la decisión se hace en la estructura del grafo mismo.
    
    Retorna el nombre del siguiente nodo según la condición.
    """
    if tiene_error_fatal(state):
        # Error fatal → saltar directamente a generar respuesta de error
        return "generador_respuesta"

    # Sin error fatal → ejecutar las consultas normalmente
    return "ejecutor_consultas"


def construir_grafo():
    """Construye y compila el grafo del agente.
    
    compile() convierte el grafo en un objeto ejecutable
    que puede ser llamado con invoke() o ainvoke().
    """
    grafo = StateGraph(AgentState)

    # --- Registrar nodos ---
    grafo.add_node("clasificador", clasificar_pregunta)
    grafo.add_node("ejecutor_consultas", ejecutar_consultas)
    grafo.add_node("generador_respuesta", generar_respuesta)

    # --- Arista de entrada: START → clasificador ---
    grafo.add_edge(START, "clasificador")

    # --- Arista condicional después del clasificador ---
    # La función decidir_después_de_clasificador retorna el nombre
    # del siguiente nodo según si hay error fatal o no.
    grafo.add_conditional_edges(
        "clasificador",
        decidir_después_de_clasificador
    )

    # --- Arista directa: ejecutor → generador ---
    # Después de las consultas, siempre generamos respuesta
    grafo.add_edge("ejecutor_consultas", "generador_respuesta")

    # --- Arista final: generador → END ---
    grafo.add_edge("generador_respuesta", END)

    return grafo.compile()


# Instancia única del grafo compilado.
# Se crea cuando este módulo se importa por primera vez.
agent = construir_grafo()