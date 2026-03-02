"""Nodo ejecutor de SQL dinámico.

Ejecuta la consulta SQL generada por el LLM contra la base de datos.
Reemplaza al query_executor.py que despachaba queries hardcodeadas.

SEGURIDAD:
- Establece statement_timeout de 5 segundos por query.
- Extrae columnas dinámicamente de cursor.description.
- En caso de error, guarda el mensaje para el loop de reintento.
"""

from app.state import AgentState
from utils.database import get_connection


def _inferir_fuente(sql: str) -> str:
    """Infiere un nombre legible de la fuente desde las tablas en el SQL."""
    sql_lower = sql.lower()
    if "inventario" in sql_lower:
        return "inventario"
    elif "garantias" in sql_lower:
        return "garantias"
    elif "movimientos_tecnicos" in sql_lower:
        return "movimientos_tecnicos"
    elif "solicitudes" in sql_lower:
        return "solicitudes"
    elif "registro_conteo" in sql_lower or "detalles_conteo" in sql_lower:
        return "conteos"
    elif "repuestos" in sql_lower:
        return "repuestos"
    return "consulta"


async def ejecutar_sql(state: AgentState) -> dict:
    """Ejecuta la query SQL del estado contra la base de datos.

    Retorna datos en el formato que espera response_generator.py:
    {"fuente": "nombre", "datos": [lista_de_dicts]}

    Si falla, guarda el error en sql_error_anterior para que
    el generador pueda autocorregir en el siguiente intento.
    """
    sql = state.sql_generado

    if not sql:
        return {
            "errores": [{
                "nodo": "ejecutor_sql",
                "mensaje": "No hay SQL para ejecutar",
                "recuperable": True
            }]
        }

    print(f"EJECUTOR_SQL - Ejecutando: {sql}...")

    try:
        async with get_connection() as conn:
            # Timeout de seguridad: 5 segundos máximo por query
            await conn.execute("SET statement_timeout = '5000'")

            cursor = await conn.execute(sql)
            rows = await cursor.fetchall()

            # Extraer nombres de columnas dinámicamente
            if cursor.description:
                columnas = [desc[0] for desc in cursor.description]
            else:
                columnas = []

            datos = [dict(zip(columnas, row)) for row in rows]

            # Restaurar timeout por defecto
            await conn.execute("RESET statement_timeout")

        fuente = _inferir_fuente(sql)
        print(f"EJECUTOR_SQL - {fuente}: {len(datos)} filas retornadas")

        return {
            "contexto_db": [{"fuente": fuente, "datos": datos}],
            "sql_error_anterior": "",  # Limpiar error si la ejecución fue exitosa
        }

    except Exception as e:
        error_msg = str(e)[:300]
        print(f"EJECUTOR_SQL - ERROR: {error_msg}")

        return {
            "sql_error_anterior": error_msg,
            "errores": [{
                "nodo": "ejecutor_sql",
                "mensaje": f"Error ejecutando SQL: {error_msg}",
                "recuperable": True
            }]
        }
