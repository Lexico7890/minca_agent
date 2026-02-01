"""Nodo clasificador del grafo.

Este nodo analiza la pregunta del usuario y determina:
1. Qué categorías de datos necesita consultar (intenciones)
2. Qué tipo de operación es (lectura, escritura, etc.)

Usa Gemini directamente sin LangChain. El modelo retorna
JSON estructurado que validamos con Pydantic.
"""

import json
from pydantic import BaseModel, ValidationError
from app.state import AgentState, TipoOperacion
from utils.gemini import gemini


# --- Schema de validación para la respuesta del modelo ---

class ClasificacionResult(BaseModel):
    """Estructura que expect el modelo que retorne.
    
    Usamos Pydantic para validar que el JSON del modelo
    tiene exactamente la forma que esperamos. Si no,
    capturamos el error en lugar de crashear.
    """
    intenciones: list[str]
    tipo_operacion: str


# --- Prompt del clasificador ---

SYSTEM_PROMPT = """Eres un clasificador de intenciones para un sistema de inventario industrial llamado Minca Electric.

Tu trabajo es analizar la pregunta del usuario y retornar ÚNICAMENTE un objeto JSON válido, sin texto adicional, sin explicaciones, sin bloques de código.

Las categorías de información disponibles son:
- "inventario": cantidades, posiciones, stock de repuestos en localizaciones
- "garantias": garantías de repuestos, estados (pendiente, resuelta), motivos de falla
- "movimientos_tecnicos": movimientos de repuestos realizados por técnicos, órdenes de trabajo
- "solicitudes": solicitudes de repuestos entre localizaciones, trazabilidad, estados
- "conteos": auditorías físicas, conteos, diferencias encontradas
- "repuestos": información del catálogo de repuestos (referencias, marcas, descripciones)

Los tipos de operación son:
- "lectura": el usuario solo quiere información
- "insertar": el usuario quiere agregar datos nuevos
- "actualizar": el usuario quiere modificar datos existentes
- "eliminar": el usuario quiere borrar datos

REGLAS:
- Si la pregunta requiere información de varias categorías, incluye todas.
- Si no entiendes la pregunta, retorna intenciones: ["no_reconocida"].
- Por ahora el sistema solo soporta lectura. Si el usuario pide escribir, igualmente detecta la intención correcta pero marca tipo_operacion como lo que detectas (lo manejamos después).

Retorna SOLO esto (sin comillas extras, sin markdown):
{"intenciones": ["categoria1", "categoria2"], "tipo_operacion": "lectura"}

Ejemplos:
- "¿Cuántos filtros hay en la bodega?" → {"intenciones": ["inventario"], "tipo_operacion": "lectura"}
- "¿Cuál es el estado de la garantía del repuesto ABC?" → {"intenciones": ["garantias"], "tipo_operacion": "lectura"}
- "Dame el stock y las garantías pendientes" → {"intenciones": ["inventario", "garantias"], "tipo_operacion": "lectura"}
- "Agrega un nuevo repuesto" → {"intenciones": ["repuestos"], "tipo_operacion": "insertar"}
- "¿Qué es lo que hace?" → {"intenciones": ["no_reconocida"], "tipo_operacion": "lectura"}
"""


def clasificar_pregunta(state: AgentState) -> AgentState:
    """Nodo clasificador: analiza la pregunta y determina intenciones y tipo de operación.
    
    Manejo de errores:
    - Si Gemini falla completamente → error no recuperable, el grafo no puede seguir.
    - Si el modelo retorna JSON inválido → intentamos parsear de forma flexible.
    - Si la validación Pydantic falla → marcamos como no_reconocida en lugar de crashear.
    """
    try:
        # Construir el contexto de memoria para el clasificador.
        # Si hay conversación previa, la incluimos para que el clasificador
        # entienda referencias contextuales.
        contexto_memoria = ""
        if state.memoria:
            # Tomamos los últimos 4 mensajes como contexto
            mensajes_recientes = state.memoria[-4:]
            contexto_memoria = "\n\nContexto de la conversación previa:\n"
            for msg in mensajes_recientes:
                rol = "Usuario" if msg.rol == "usuario" else "Agente"
                contexto_memoria += f"{rol}: {msg.contenido}\n"

        prompt = f"{contexto_memoria}\nPregunta actual del usuario: {state.pregunta_actual}"

        # Llamar a Gemini con temperatura baja (clasificación debe ser determinista)
        respuesta_texto = gemini.llamar(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperatura=0.1,
            max_tokens=256
        )

        # --- Parseo flexible del JSON ---
        # El modelo a veces agrega bloques de markdown o espacios extra.
        # Limpiamos antes de parsear.
        texto_limpio = respuesta_texto.strip()
        if texto_limpio.startswith("```"):
            # Remover bloques de código markdown si los agrega
            líneas = texto_limpio.split("\n")
            texto_limpio = "\n".join(
                l for l in líneas
                if not l.strip().startswith("```")
            ).strip()

        datos = json.loads(texto_limpio)

        # --- Validación con Pydantic ---
        clasificacion = ClasificacionResult(**datos)

        # Asignar al estado
        state.intenciones = clasificacion.intenciones
        state.tipo_operacion = TipoOperacion(clasificacion.tipo_operacion)

    except json.JSONDecodeError:
        # El modelo no retornó JSON válido.
        # No es un error fatal, simplemente no entendimos la pregunta.
        state.intenciones = ["no_reconocida"]
        state.errores.append({
            "nodo": "clasificador",
            "mensaje": "El modelo no retornó JSON válido en la clasificación",
            "recuperable": True
        })

    except ValidationError as e:
        # El JSON es válido pero no tiene la estructura esperada.
        state.intenciones = ["no_reconocida"]
        state.errores.append({
            "nodo": "clasificador",
            "mensaje": f"Estructura de clasificación inválida: {str(e)}",
            "recuperable": True
        })

    except (ValueError, KeyError) as e:
        # Tipo de operación no reconocido u otro error de datos.
        state.intenciones = ["no_reconocida"]
        state.errores.append({
            "nodo": "clasificador",
            "mensaje": f"Error procesando clasificación: {str(e)}",
            "recuperable": True
        })

    except RuntimeError as e:
        # Error de la API de Gemini. Este NO es recuperable
        # porque sin clasificación no sabemos qué consultar.
        state.errores.append({
            "nodo": "clasificador",
            "mensaje": str(e),
            "recuperable": False
        })

    return state