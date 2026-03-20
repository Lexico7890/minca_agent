"""Nodo generador de SQL dinámico.

Este nodo reemplaza al clasificador y al despacho de queries hardcodeadas.
Recibe la pregunta del usuario y genera una consulta SQL precisa
usando el LLM con el schema de la DB como contexto.

SEGURIDAD:
- El LLM solo puede generar consultas SELECT.
- Toda query se valida antes de ejecutarse.
- Se fuerza un LIMIT máximo en queries no-agregadas.
- Se bloquean keywords peligrosos (INSERT, DROP, etc.).
"""

import json
import re
from app.state import AgentState
from utils.gemini import gemini


# --- Schema compacto de la base de datos (~250 tokens) ---

SCHEMA_CONTEXT = """T=timestamp B=bool I=int J=json ?=nullable
repuestos(id_repuesto,referencia,nombre,marca,tipo,descripcion,descontinuado B)
localizacion(id_localizacion,nombre)
usuarios(id_usuario,nombre,email,activo B,aprobado B,id_rol>roles)
roles(id_rol,nombre,permissions J)
inventario(id_repuesto>repuestos,id_localizacion>localizacion,cantidad I,cantidad_minima I,posicion,nuevo_hasta T?)
garantias(id_garantia,id_repuesto>repuestos,estado,motivo_falla,comentarios_resolucion,orden,solicitante,kilometraje I,id_localizacion>localizacion,id_usuario_reporta>usuarios,id_tecnico_asociado>usuarios?,created_at T,updated_at T)
movimientos_tecnicos(id_repuesto>repuestos,concepto,tipo,cantidad I,numero_orden,descargada B,id_localizacion>localizacion,id_usuario_responsable>usuarios,id_tecnico_asignado>usuarios,fecha T)
registro_conteo(id_conteo,tipo,id_localizacion>localizacion,id_usuario>usuarios,total_items_auditados I,total_diferencia_encontrada I,total_items_pq I,observaciones,created_at T)
detalles_conteo(id_conteo>registro_conteo,id_repuesto>repuestos,cantidad_sistema I,cantidad_csa I,diferencia I,cantidad_pq I)
usuarios_localizacion(id_usuario>usuarios,id_localizacion>localizacion)

garantias.estado:'Sin enviar'|'Pendiente'|'Aprobada'|'Rechazada'
solicitudes.estado:pendiente|alistada|despachada|recibida
registro_conteo.tipo:total|parcial"""


# --- Prompt del generador de SQL ---

SQL_SYSTEM_PROMPT = """Eres un generador de consultas SQL para PostgreSQL en un sistema de inventario industrial (Trazea Management System).
Convierte preguntas en español a consultas SQL precisas.

REGLAS:
1. SOLO genera SELECT. Nunca INSERT, UPDATE, DELETE, DROP ni otra operación.
2. NUNCA uses frases completas en un solo ILIKE. Separa CADA palabra clave en su propio ILIKE con AND.
3. Trunca cada palabra al tronco (quita s, es, as del final) para matchear plural y singular. Ejemplo: "pastillas" → '%pastill%', "hidráulicas" → '%hidraulic%', "llantas" → '%llant%', "filtros" → '%filtr%'.
4. Usa el texto EXACTO que el usuario escribió como base para los keywords. No corrijas ortografía ni uses sinónimos.
5. Siempre incluye r.nombre en el SELECT para que el usuario vea qué repuesto se encontró.
6. Incluye JOINs para mostrar nombres legibles (no UUIDs).
7. Para cantidades totales usa SUM(), COUNT() u otra función de agregación, siempre con GROUP BY r.nombre.
8. Convierte UUIDs a texto con ::text cuando los incluyas en SELECT.
9. Usa alias descriptivos (AS nombre_legible).
10. Agrega LIMIT 50 excepto en agregaciones puras (COUNT/SUM sin GROUP BY).

{schema}

FORMATO: Retorna SOLO un JSON válido, sin markdown ni texto extra:
{{"sql": "SELECT ...", "explicacion": "descripción breve"}}

EJEMPLOS:
P: "cuantas pastillas hidráulicas hay en tester location"
{{"sql": "SELECT r.nombre, SUM(i.cantidad) AS total FROM inventario i JOIN repuestos r ON i.id_repuesto = r.id_repuesto JOIN localizacion l ON i.id_localizacion = l.id_localizacion WHERE r.nombre ILIKE '%pastill%' AND r.nombre ILIKE '%hidraulic%' AND l.nombre ILIKE '%tester%' GROUP BY r.nombre", "explicacion": "Pastillas hidráulicas en Tester Location"}}

P: "cuantas llantas sellomatic tenemos en bodega central"
{{"sql": "SELECT r.nombre, SUM(i.cantidad) AS total FROM inventario i JOIN repuestos r ON i.id_repuesto = r.id_repuesto JOIN localizacion l ON i.id_localizacion = l.id_localizacion WHERE r.nombre ILIKE '%llant%' AND r.nombre ILIKE '%sellomatic%' AND l.nombre ILIKE '%bodega central%' GROUP BY r.nombre", "explicacion": "Llantas sellomatic en Bodega Central"}}

P: "qué repuestos tienen stock bajo"
{{"sql": "SELECT r.referencia, r.nombre, i.cantidad, i.cantidad_minima, l.nombre AS localizacion FROM inventario i JOIN repuestos r ON i.id_repuesto = r.id_repuesto JOIN localizacion l ON i.id_localizacion = l.id_localizacion WHERE i.cantidad <= i.cantidad_minima AND i.cantidad > 0 ORDER BY i.cantidad LIMIT 50", "explicacion": "Repuestos con stock bajo"}}

P: "cuántas garantías pendientes hay"
{{"sql": "SELECT COUNT(*) AS total_pendientes FROM garantias WHERE estado ILIKE '%pendiente%'", "explicacion": "Total de garantías pendientes"}}

P: "muéstrame los filtros de aceite en taller norte"
{{"sql": "SELECT r.referencia, r.nombre, i.cantidad, i.posicion, l.nombre AS localizacion FROM inventario i JOIN repuestos r ON i.id_repuesto = r.id_repuesto JOIN localizacion l ON i.id_localizacion = l.id_localizacion WHERE r.nombre ILIKE '%filtr%' AND r.nombre ILIKE '%aceit%' AND l.nombre ILIKE '%taller norte%' ORDER BY r.nombre LIMIT 50", "explicacion": "Filtros de aceite en Taller Norte"}}"""


# --- Keywords prohibidos en SQL ---

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE",
    "SET", "MERGE", "COPY",
]


# --- Funciones de validación ---

def validar_sql(sql: str) -> str | None:
    """Valida que el SQL sea una consulta SELECT segura.

    Retorna None si es válido, o un string con el error si no lo es.
    """
    if not sql:
        return "SQL vacío"

    normalized = sql.strip().upper()

    # Debe empezar con SELECT o WITH (para CTEs)
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return "La consulta debe comenzar con SELECT"

    # Buscar keywords prohibidos como palabras completas
    for keyword in FORBIDDEN_KEYWORDS:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, normalized):
            return f"Keyword prohibido detectado: {keyword}"

    # Detectar múltiples statements (ignorando contenido dentro de strings)
    sql_sin_strings = re.sub(r"'[^']*'", "", sql)
    if ";" in sql_sin_strings:
        return "Múltiples statements detectados"

    # Detectar comentarios SQL
    if "--" in sql_sin_strings or "/*" in sql_sin_strings:
        return "Comentarios SQL no permitidos"

    return None


def _enforce_limit(sql: str, max_limit: int = 50) -> str:
    """Asegura que la query tenga LIMIT. Lo agrega si falta, lo reduce si excede.

    No agrega LIMIT a agregaciones puras (COUNT/SUM/AVG sin GROUP BY).
    """
    normalized = sql.strip().upper()

    # Detectar agregación pura (sin GROUP BY)
    agg_funcs = ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
    has_agg = any(f in normalized for f in agg_funcs)
    is_pure_agg = has_agg and "GROUP BY" not in normalized

    if is_pure_agg:
        return sql

    # Verificar si ya tiene LIMIT al final de la consulta
    limit_match = re.search(r'\bLIMIT\s+(\d+)\s*;?\s*$', normalized)
    if limit_match:
        existing = int(limit_match.group(1))
        if existing > max_limit:
            sql = re.sub(
                r'\bLIMIT\s+\d+\s*;?\s*$',
                f'LIMIT {max_limit}',
                sql,
                flags=re.IGNORECASE
            )
        return sql

    # Si hay un LIMIT intermedio pero no al final, agregamos uno global
    # No tiene LIMIT al final, agregarlo
    return sql.rstrip().rstrip(";") + f" LIMIT {max_limit}"


def _inferir_intenciones(sql: str) -> list[str]:
    """Infiere categorías de intención desde las tablas en el SQL.

    Mantiene compatibilidad con el response generator y main.py
    que esperan el campo intenciones.
    """
    sql_lower = sql.lower()
    intenciones = []
    table_map = {
        "inventario": "inventario",
        "garantias": "garantias",
        "movimientos_tecnicos": "movimientos_tecnicos",
        "solicitudes": "solicitudes",
        "registro_conteo": "conteos",
        "detalles_conteo": "conteos",
        "repuestos": "repuestos",
        "localizacion": "localizacion",
        "usurios": "usurios",
        "roles": "roles",
        "usuarios_localizacion": "usuarios_localizacion"
    }
    for table, intencion in table_map.items():
        if table in sql_lower and intencion not in intenciones:
            intenciones.append(intencion)
    return intenciones or ["consulta_general"]


def _es_saludo(pregunta: str) -> bool:
    """Detecta si la pregunta es un saludo simple."""
    saludos = ["hola", "buenos días", "buenas tardes", "hey", "qué tal"]
    return any(s in pregunta.lower() for s in saludos) and len(pregunta.split()) <= 5


# --- Nodo principal del generador ---

def generar_sql(state: AgentState) -> dict:
    """Nodo generador de SQL: convierte la pregunta en una consulta SQL.

    En caso de reintento, incluye el error anterior para que el LLM
    pueda autocorregir su query.
    """

    # Saludos no necesitan SQL
    if _es_saludo(state.pregunta_actual):
        return {"intenciones": ["saludo"]}

    try:
        # Construir prompt con contexto de memoria
        prompt_parts = []

        if state.memoria:
            mensajes_recientes = state.memoria[-4:]
            contexto = "\n".join(
                f"{'Usuario' if m.rol == 'usuario' else 'Agente'}: {m.contenido}"
                for m in mensajes_recientes
            )
            prompt_parts.append(f"Conversación previa:\n{contexto}")

        prompt_parts.append(f"Pregunta: {state.pregunta_actual}")

        # Si es reintento, incluir error para autocorrección
        if state.sql_error_anterior:
            prompt_parts.append(
                f"\nINTENTO ANTERIOR FALLÓ.\n"
                f"Error: {state.sql_error_anterior}\n"
                f"SQL anterior: {state.sql_generado}\n"
                f"Corrige la consulta."
            )

        prompt = "\n\n".join(prompt_parts)
        system = SQL_SYSTEM_PROMPT.format(schema=SCHEMA_CONTEXT)

        # Llamar al LLM con temperatura baja para precisión
        respuesta_texto = gemini.llamar(
            prompt=prompt,
            system_prompt=system,
            temperatura=0.1,
            max_tokens=512,
            use_quality_model=True
        )

        # Parseo flexible del JSON
        texto_limpio = respuesta_texto.strip()
        if texto_limpio.startswith("```"):
            lineas = texto_limpio.split("\n")
            texto_limpio = "\n".join(
                l for l in lineas if not l.strip().startswith("```")
            ).strip()

        datos = json.loads(texto_limpio)
        sql = datos.get("sql", "").strip()
        explicacion = datos.get("explicacion", "")

        print(f"GENERADOR_SQL - SQL: {sql[:100]}...")
        print(f"GENERADOR_SQL - Explicación: {explicacion}")

        # Validar seguridad del SQL
        error_validacion = validar_sql(sql)
        if error_validacion:
            print(f"GENERADOR_SQL - Validación falló: {error_validacion}")
            return {
                "sql_generado": sql,
                "intenciones": ["no_reconocida"],
                "errores": [{
                    "nodo": "generador_sql",
                    "mensaje": f"SQL inválido: {error_validacion}",
                    "recuperable": False
                }]
            }

        # Forzar LIMIT de seguridad
        sql = _enforce_limit(sql, max_limit=50)

        # Construir resultado
        result = {
            "sql_generado": sql,
            "sql_explicacion": explicacion,
            "intenciones": _inferir_intenciones(sql),
            "sql_error_anterior": "",  # Limpiar error anterior si hubo éxito
        }
        print("result SQL: ", result)

        # Si es reintento, decrementar contador
        if state.sql_error_anterior:
            result["sql_reintentos"] = state.sql_reintentos - 1

        return result

    except json.JSONDecodeError:
        print("GENERADOR_SQL - Error: JSON inválido del LLM")
        return {
            "intenciones": ["no_reconocida"],
            "errores": [{
                "nodo": "generador_sql",
                "mensaje": "El LLM no retornó JSON válido",
                "recuperable": True
            }]
        }

    except RuntimeError as e:
        print(f"GENERADOR_SQL - Error API: {str(e)[:100]}")
        return {
            "errores": [{
                "nodo": "generador_sql",
                "mensaje": str(e),
                "recuperable": False
            }]
        }
