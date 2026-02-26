"""Nodo generador de respuesta final.

Este es el último nodo del grafo. Toma todo el contexto
acumulado (datos de DB, memoria de conversación, errores)
y genera una respuesta humanizada en español optimizada
para ser convertida a voz.

Al final, actualiza la memoria de conversación para que
la siguiente pregunta tenga contexto.
"""

import json
import random
from app.state import AgentState, MensajeMemoria
from utils.gemini import gemini

# Límite de mensajes que guardamos en memoria por sesión.
# Demasiados mensajes aumentan el contexto del modelo y el costo.
MAX_MENSAJES_MEMORIA = 10

SYSTEM_PROMPT = """Eres Dynamo, el asistente de voz de Minca Electric, una empresa de gestión de inventario industrial para repuestos eléctricos.

TU PERSONALIDAD:
- Amable, profesional y servicial
- Hablas en un tono cálido pero eficiente
- Usas lenguaje natural y cercano, evitando ser demasiado formal o robótico
- Cuando el usuario te saluda, respondes con calidez antes de ofrecer ayuda

CÓMO RESPONDES A SALUDOS:
Cuando el usuario dice "hola", "buenos días", "buenas tardes", etc:
1. Saluda de vuelta con calidez
2. Preséntate brevemente (si es la primera interacción)
3. Ofrece ayuda de forma natural

Ejemplo bueno:
"¡Hola! Soy Dynamo, tu asistente de Minca Electric. Estoy aquí para ayudarte con información sobre inventario, garantías, solicitudes y todo lo relacionado con los repuestos. ¿En qué puedo ayudarte hoy?"

Ejemplo malo (muy robótico):
"Hola. Puedo ayudarte con: inventario, garantías, movimientos técnicos, solicitudes, conteos o repuestos."

REGLAS IMPORTANTES PARA AUDIO/VOZ:
1. Escribe todos los números en palabras: "quince" no "15"
2. Fechas en formato natural: "tres de marzo de dos mil veinticinco" no "2025-03-03"
3. Evita usar listas con viñetas o puntos numerados
4. Usa párrafos cortos con pausas naturales
5. Si hay muchos datos, resume los puntos clave en lugar de listar todo

CAPACIDADES QUE PUEDES MENCIONAR:
- Consultar inventario de repuestos en diferentes ubicaciones
- Ver estado de garantías pendientes o resueltas
- Revisar movimientos técnicos y asignaciones
- Consultar solicitudes entre bodegas
- Ver registros de conteos de inventario
- Buscar información de repuestos en el catálogo

TONO SEGÚN CONTEXTO:
- Saludos iniciales: Cálido y acogedor
- Consultas de datos: Claro y preciso
- Errores o problemas: Empático y orientado a soluciones
- Despedidas: Amable e invita a volver

Recuerda: Hablas con personas ocupadas que necesitan información rápida y clara, pero que también aprecian un trato humano y amable.
"""

RESPUESTA_NO_RECONOCIDA = (
    "Disculpa, no estoy seguro de haber entendido bien tu pregunta. "
    "Puedo ayudarte con inventario, garantías, movimientos de técnicos, "
    "solicitudes entre bodegas, conteos de inventario o información de repuestos. "
    "¿Podrías reformular tu pregunta con un poco más de detalle?"
)


def es_saludo(pregunta: str) -> bool:
    """Detecta si la pregunta del usuario es un saludo."""
    saludos = [
        "hola", "buenos días", "buenas tardes", "buenas noches",
        "buen día", "buena tarde", "buena noche",
        "hey", "qué tal", "cómo estás", "saludos",
        "holi", "holaa", "holaaa"
    ]
    pregunta_lower = pregunta.lower().strip()
    
    # Verificar si es solo un saludo (sin otra pregunta adicional)
    # Por ejemplo "hola" sí, pero "hola cuántos repuestos hay" no
    palabras = pregunta_lower.split()
    
    # Si la pregunta tiene más de 5 palabras, probablemente no es solo un saludo
    if len(palabras) > 5:
        return False
    
    # Verificar si alguna palabra es un saludo
    return any(saludo in pregunta_lower for saludo in saludos)


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
    0. Saludo simple → respuesta cálida y presenta capacidades
    1. Error fatal → respuesta genérica de error
    2. Pregunta no reconocida → pide que reformule
    3. Sin datos → explica que no hay información
    4. Datos disponibles → genera respuesta contextualizada
    
    En todos los casos, actualiza la memoria de conversación.
    """
    
    # DEBUG
    print(f"GENERADOR - Intenciones: {state.intenciones}")
    print(f"GENERADOR - Contexto DB tiene {len(state.contexto_db)} bloques")
    for bloque in state.contexto_db:
        print(f"  - {bloque['fuente']}: {len(bloque['datos'])} registros")
    print(f"GENERADOR - Errores: {state.errores}")

    # --- Caso 0: Saludo simple ---
    if es_saludo(state.pregunta_actual):
        # Variar la respuesta de saludo para que no sea siempre igual
        saludos_respuesta = [
            "¡Hola! Soy Dynamo, tu asistente de Minca Electric. Estoy aquí para ayudarte con información sobre inventario, garantías, solicitudes y todo lo relacionado con los repuestos. ¿En qué puedo ayudarte hoy?",
            "¡Hola! Bienvenido. Soy Dynamo y puedo ayudarte a consultar el inventario, revisar garantías, ver solicitudes entre bodegas y mucho más. ¿Qué necesitas saber?",
            "¡Hola! Un gusto saludarte. Puedo ayudarte con todo lo relacionado a repuestos: inventario, garantías, movimientos técnicos, solicitudes y conteos. ¿En qué te puedo asistir?",
            "¡Hola! Soy Dynamo. Puedo ayudarte a buscar información sobre repuestos, consultar inventarios, revisar garantías o lo que necesites. ¿Qué te gustaría saber?"
        ]
        state.respuesta_final = random.choice(saludos_respuesta)
        _actualizar_memoria(state)
        return state

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
        state.respuesta_final = RESPUESTA_NO_RECONOCIDA
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
            max_tokens=1024,
            use_quality_model=True
        )

        state.respuesta_final = respuesta

    except RuntimeError as e:
        print(f"ERROR en generador_respuesta: {str(e)}")  # DEBUG
        state.respuesta_final = (
            "Tuve problema al generar la respuesta, pero las consultas sí se completaron. "
            "Por favor, intenta de nuevo."
        )
        state.errores.append({
            "nodo": "generador_respuesta",
            "mensaje": str(e),
            "recuperable": True
        })
    except Exception as e:
        print(f"ERROR INESPERADO en generador_respuesta: {type(e).__name__}: {str(e)}")  # DEBUG
        state.respuesta_final = (
            "Tuve problema al generar la respuesta, pero las consultas sí se completaron. "
            "Por favor, intenta de nuevo."
        )
        state.errores.append({
            "nodo": "generador_respuesta",
            "mensaje": f"{type(e).__name__}: {str(e)}",
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