"""Nodo generador de respuesta final.

Este es el último nodo del grafo. Toma todo el contexto
acumulado (datos de DB, memoria de conversación, errores)
y genera una respuesta humanizada en español optimizada
para ser convertida a voz.

Al final, actualiza la memoria de conversación para que
la siguiente pregunta tenga contexto.
"""

import json
from app.state import AgentState, MensajeMemoria
from utils.gemini import gemini

# Límite de mensajes que guardamos en memoria por sesión.
# Demasiados mensajes aumentan el contexto del modelo y el costo.
MAX_MENSAJES_MEMORIA = 10

SYSTEM_PROMPT = """Eres un asistente de voz profesional para Minca Electric, una empresa industrial.
Tu respuesta se va a convertir en audio mediante text-to-speech, por lo que debe sonar natural al escucharse.

REGLAS DE FORMATO:
- Responde siempre en español.
- Sé conciso pero completo. Evita listas largas porque no se ven en audio.
- Los números escríbelos en palabras cuando es natural: "hay quince unidades" en lugar de "15".
  Excepto para referencias de repuestos como "ABC-123", esas se dejan tal cual.
- Las fechas sintetízalas de forma natural: "el tres de marzo" en lugar de "2025-03-03T00:00:00".
- No uses viñetas, guiones ni numeración. Escribe en prosa.
- Si hay muchos datos, prioriza los más relevantes y menciona que hay más disponibles.

REGLAS DE CONTENIDO:
- Responde ÚNICAMENTE con información que esté en los datos proporcionados.
- No inventes datos ni hagas suposiciones.
- Si los datos están vacíos, dilo claramente y sugiere reformular.
- Si hay errores parciales (algunas consultas fallaron), menciona qué información no pudo obtenerse.
- Si el usuario hace una referencia a algo anterior en la conversación, úsala para contextualizar.
"""


def tiene_error_fatal(state: AgentState) -> bool:
    """Verifica si hay algún error no recuperable."""
    return any(not e["recuperable"] for e in state.errores)


def construir_contexto_memoria(state: AgentState) -> str:
    """Construye el texto de memoria para incluir en el prompt.
    
    Si hay conversación previa, la formatea de forma que el modelo
    entienda el contexto sin necesidad de re-preguntar.
    """
    if not state.memoria:
        return ""

    mensajes_recientes = state.memoria[-MAX_MENSAJES_MEMORIA:]
    texto = "\n--- Historial de la conversación actual ---\n"
    for msg in mensajes_recientes:
        rol = "Usuario" if msg.rol == "usuario" else "Asistente"
        texto += f"{rol}: {msg.contenido}\n"
    texto += "--- Fin del historial ---\n"
    return texto


def construir_contexto_datos(state: AgentState) -> str:
    """Convierte los datos de las consultas en texto que el modelo puede leer."""
    if not state.contexto_db and not state.contexto_rag:
        return "No se encontraron datos en la base de datos."

    partes = []

    if state.contexto_db:
        partes.append("=== Datos de la base de datos ===")
        for bloque in state.contexto_db:
            partes.append(f"\n[{bloque['fuente']}]")
            if bloque["datos"]:
                # Formatear como JSON legible para el modelo
                partes.append(json.dumps(bloque["datos"], ensure_ascii=False, indent=2))
            else:
                partes.append("No hay datos en esta categoría.")

    if state.contexto_rag:
        partes.append("\n=== Datos de documentos ===")
        for bloque in state.contexto_rag:
            partes.append(f"\n[{bloque['fuente']}]")
            partes.append(json.dumps(bloque["datos"], ensure_ascii=False, indent=2))

    return "\n".join(partes)


def construir_nota_errores(state: AgentState) -> str:
    """Si hubo errores recuperables, le informa al modelo para que los mencione."""
    errores_recuperables = [e for e in state.errores if e["recuperable"]]
    if not errores_recuperables:
        return ""

    fuentes_fallidas = [e["nodo"] for e in errores_recuperables]
    return (
        f"\nNOTA: Las siguientes consultas tuvieron problemas y no retornaron datos: "
        f"{', '.join(fuentes_fallidas)}. "
        f"Menciona al usuario que esa información no pudo obtenerse en este momento."
    )


def generar_respuesta(state: AgentState) -> AgentState:
    """Genera la respuesta final humanizada.
    
    Manejo de casos especiales:
    1. Error fatal → respuesta genérica de error
    2. Pregunta no reconocida → pide que reformule
    3. Sin datos → explica que no hay información
    4. Datos disponibles → genera respuesta contextualizada
    
    En todos los casos, actualiza la memoria de conversación.
    """

    # --- Caso 1: Error fatal ---
    if tiene_error_fatal(state):
        state.respuesta_final = (
            "Lamento mucho, ocurrió un problema interno al procesar tu pregunta. "
            "Por favor, intenta de nuevo en un momento."
        )
        # No actualizar memoria con errores fatales
        return state

    # --- Caso 2: Pregunta no reconocida ---
    if state.intenciones == ["no_reconocida"]:
        state.respuesta_final = (
            "No entendí del todo tu pregunta. ¿Podrías reformularla con más detalle? "
            "Puedo ayudarte con información sobre inventario, garantías, "
            "movimientos técnicos, solicitudes, conteos o repuestos."
        )
        # Actualizar memoria incluso cuando no entendemos
        _actualizar_memoria(state)
        return state

    # --- Caso 3 y 4: Generar respuesta con Gemini ---
    try:
        contexto_memoria = construir_contexto_memoria(state)
        contexto_datos = construir_contexto_datos(state)
        nota_errores = construir_nota_errores(state)

        prompt = (
            f"{contexto_memoria}\n"
            f"Pregunta actual del usuario: {state.pregunta_actual}\n\n"
            f"{contexto_datos}\n"
            f"{nota_errores}\n\n"
            f"Genera una respuesta natural basándote en estos datos."
        )

        respuesta = gemini.llamar(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperatura=0.4,  # Un poco más alto que la clasificación para sonar natural
            max_tokens=1024
        )

        state.respuesta_final = respuesta

    except RuntimeError as e:
        state.respuesta_final = (
            "Tuve problema al generar la respuesta, pero las consultas sí se completaron. "
            "Por favor, intenta de nuevo."
        )
        state.errores.append({
            "nodo": "generador_respuesta",
            "mensaje": str(e),
            "recuperable": True
        })

    # Actualizar memoria de la conversación
    _actualizar_memoria(state)

    return state


def _actualizar_memoria(state: AgentState):
    """Agrega el intercambio actual a la memoria de conversación.
    
    Agregamos tanto la pregunta del usuario como la respuesta del agente.
    Si la memoria excede el límite, eliminamos los mensajes más antiguos.
    """
    # Agregar pregunta del usuario
    state.memoria.append(MensajeMemoria(
        rol="usuario",
        contenido=state.pregunta_actual
    ))

    # Agregar respuesta del agente (si la hay)
    if state.respuesta_final:
        state.memoria.append(MensajeMemoria(
            rol="agente",
            contenido=state.respuesta_final
        ))

    # Limitar el tamaño de la memoria
    if len(state.memoria) > MAX_MENSAJES_MEMORIA:
        state.memoria = state.memoria[-MAX_MENSAJES_MEMORIA:]