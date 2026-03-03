"""Generador de respuestas con TOON correcto.

Formato según: https://github.com/toon-format/toon
Reducción: 65-70% de tokens vs JSON
"""

import random
from app.state import AgentState, MensajeMemoria
from utils.gemini import gemini

MAX_MENSAJES_MEMORIA = 2

SYSTEM_PROMPT = """Eres Dynamo, asistente de Trasea para Minca Electric.
Amable, conciso, directo. Números en palabras.
Los datos están en formato TOON (compacto)."""

SYSTEM_PROMPT_RAG = """Eres Dynamo, asistente de Trasea para Minca Electric.
Responde basándote SOLO en los fragmentos de documentos proporcionados.
Si la información no está en los documentos, dilo claramente.
Cita el documento y página cuando sea posible. Ejemplo: (Política de Garantías, p.3).
Amable, conciso, directo."""

RESPUESTA_NO_RECONOCIDA = "No entendí. Reformula tu pregunta."


def encode_toon_array(name: str, data: list, max_items: int = 8) -> str:
    """Codifica array en formato TOON correcto.
    
    Formato: name[count]{field1,field2,...}:
             value1,value2,...
    """
    if not data:
        return f"{name}[0]{{}}:"
    
    data_limited = data[:max_items]
    total = len(data)
    
    # Schema (campos del primer objeto)
    sample = data_limited[0]
    priority = ['referencia', 'nombre', 'cantidad', 'estado', 'origen', 'destino', 'localizacion']
    
    fields = [f for f in priority if f in sample]
    for f in sample.keys():
        if f not in fields and len(fields) < 7:
            if not f.startswith('id_') and not f.endswith('_at'):
                fields.append(f)
    
    if not fields:
        return f"{name}[0]{{}}:"
    
    # Header TOON
    schema = ",".join(fields)
    lines = [f"{name}[{total}]{{{schema}}}:"]
    
    # Data rows
    for row in data_limited:
        values = []
        for field in fields:
            val = row.get(field, "")
            
            if val is None or val == "":
                val_str = "-"
            elif isinstance(val, bool):
                val_str = "true" if val else "false"
            elif isinstance(val, str):
                val_str = val[:35].replace(",", ";").replace("\n", " ")
            else:
                val_str = str(val)
            
            values.append(val_str)
        
        lines.append("  " + ",".join(values))
    
    if total > max_items:
        lines.append(f"  ... (+{total - max_items} more)")
    
    return "\n".join(lines)


def construir_contexto_datos_TOON(state: AgentState) -> str:
    """Construye contexto usando TOON - formato correcto."""
    if not state.contexto_db:
        return "data: none"
    
    sections = []
    
    # Context block
    context_lines = ["context:"]
    context_lines.append(f"  query: {state.pregunta_actual[:50]}")
    if state.sql_explicacion:
        context_lines.append(f"  analysis: {state.sql_explicacion[:80]}")
    context_lines.append(f"  intents: {','.join(state.intenciones[:3])}")
    sections.append("\n".join(context_lines))
    
    # Arrays en formato TOON
    for bloque in state.contexto_db:
        fuente = bloque['fuente']
        datos = bloque.get('datos', [])
        
        if datos:
            toon_array = encode_toon_array(fuente, datos, max_items=8)
            sections.append(toon_array)
    
    return "\n".join(sections)


def construir_contexto_rag(state: AgentState) -> str:
    """Formatea chunks RAG para el prompt del LLM.

    Formato compacto: documento, página, similitud + contenido truncado.
    Limita contenido por chunk para no exceder tokens.
    """
    if not state.contexto_rag:
        return "documentos: sin resultados"

    lines = [f"documentos[{len(state.contexto_rag)}]:"]
    for i, chunk in enumerate(state.contexto_rag):
        sim = chunk.get("similitud", 0)
        lines.append(
            f"  [{i+1}] {chunk.get('documento', '?')} "
            f"(p.{chunk.get('pagina', '?')}, sim:{sim:.0%})"
        )
        # Limitar contenido a ~300 chars por chunk para controlar tokens
        contenido = chunk.get("contenido", "")[:300]
        lines.append(f"  {contenido}")

    return "\n".join(lines)


def es_saludo(pregunta: str) -> bool:
    saludos = ["hola", "buenos días", "buenas tardes", "hey", "qué tal"]
    return any(s in pregunta.lower() for s in saludos) and len(pregunta.split()) <= 5


def tiene_error_fatal(state: AgentState) -> bool:
    return any(not e["recuperable"] for e in state.errores)


def construir_contexto_memoria(state: AgentState) -> str:
    if not state.memoria:
        return ""
    
    recientes = state.memoria[-(MAX_MENSAJES_MEMORIA * 2):]
    return "\n".join([f"{m.rol[0]}:{m.contenido[:60]}" for m in recientes])


def generar_respuesta(state: AgentState) -> AgentState:
    """Genera respuesta usando TOON (SQL) o contexto RAG (documentos)."""

    print(f"GENERADOR - Modo: {state.modo}, Intenciones: {state.intenciones}")
    print(f"GENERADOR - Bloques DB: {len(state.contexto_db)}, Chunks RAG: {len(state.contexto_rag)}")

    # Saludo
    if es_saludo(state.pregunta_actual):
        state.respuesta_final = random.choice([
            "¡Hola! Soy Dynamo. ¿En qué puedo ayudarte?",
            "¡Hola! ¿Qué necesitas?"
        ])
        _actualizar_memoria(state)
        return state

    # Error fatal
    if tiene_error_fatal(state):
        state.respuesta_final = "Error. Intenta nuevamente."
        return state

    # No reconocida
    if state.intenciones == ["no_reconocida"]:
        state.respuesta_final = RESPUESTA_NO_RECONOCIDA
        _actualizar_memoria(state)
        return state

    # --- Modo RAG: sin resultados ---
    if state.modo == "rag" and not state.contexto_rag:
        state.respuesta_final = (
            "No encontré información relevante en los documentos. "
            "Intenta reformular tu pregunta o verifica que el documento haya sido cargado."
        )
        _actualizar_memoria(state)
        return state

    # --- Modo SQL: ejecutado pero sin resultados ---
    if state.modo == "sql" and state.sql_generado and not state.contexto_db:
        state.respuesta_final = (
            "No encontré resultados para tu consulta. "
            "Intenta reformular la pregunta o verificar los nombres."
        )
        _actualizar_memoria(state)
        return state

    # --- Seleccionar contexto y prompt según modo ---
    if state.contexto_rag:
        datos = construir_contexto_rag(state)
        system_prompt = SYSTEM_PROMPT_RAG
        print(f"GENERADOR - Usando contexto RAG ({len(state.contexto_rag)} chunks)")
    else:
        datos = construir_contexto_datos_TOON(state)
        system_prompt = SYSTEM_PROMPT
        print(f"GENERADOR - Usando contexto TOON (SQL)")

    # Generar respuesta con LLM
    try:
        memoria = construir_contexto_memoria(state)

        # Prompt compacto
        prompt_parts = []
        if memoria:
            prompt_parts.append(f"history:\n{memoria}")
        prompt_parts.append(f"question: {state.pregunta_actual}")
        prompt_parts.append(f"\n{datos}")

        prompt = "\n".join(prompt_parts)

        # Estimar tokens
        tokens_est = len(prompt) // 4
        print(f"GENERADOR - Tokens estimados: ~{tokens_est}")

        respuesta = gemini.llamar(
            prompt=prompt,
            system_prompt=system_prompt,
            temperatura=0.4,
            max_tokens=350,
            use_quality_model=True
        )

        state.respuesta_final = respuesta

    except RuntimeError as e:
        error_msg = str(e)[:150]
        print(f"GENERADOR - ERROR: {error_msg}")

        if "rate_limit" in error_msg.lower() or "tokens" in error_msg.lower():
            state.respuesta_final = "Demasiadas consultas. Intenta en un momento."
        else:
            state.respuesta_final = "Error. Intenta de nuevo."

        state.errores.append({
            "nodo": "generador",
            "mensaje": str(e),
            "recuperable": True
        })

    _actualizar_memoria(state)
    return state


def _actualizar_memoria(state: AgentState):
    state.memoria.append(MensajeMemoria(
        rol="usuario",
        contenido=state.pregunta_actual[:70]
    ))
    
    if state.respuesta_final:
        state.memoria.append(MensajeMemoria(
            rol="agente",
            contenido=state.respuesta_final[:70]
        ))
    
    if len(state.memoria) > MAX_MENSAJES_MEMORIA * 2:
        state.memoria = state.memoria[-(MAX_MENSAJES_MEMORIA * 2):]