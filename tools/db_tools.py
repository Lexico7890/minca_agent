"""Herramientas de consulta a PostgreSQL.

Cada función corresponde a una categoría del clasificador.
Estas son las queries que el agente ejecuta contra la base de datos.

REGLAS DE SEGURIDAD:
- Todas las queries están escritas por nosotros, nunca por el modelo.
- No hay interpolación de strings de entrada del usuario en el SQL.
- Cada función maneja sus propios errores de forma independiente,
  para que si una falla, las demás sigan ejecutándose.

ESTRUCTURA DE RETORNO:
- Cada función agrega un elemento a state.contexto_db con formato:
  {"fuente": "nombre_categoria", "datos": [lista_de_diccionarios]}
- Si falla, agrega un error a state.errores marcado como recuperable.
"""

from app.state import AgentState
from utils.database import get_connection


async def consultar_inventario(state: AgentState) -> AgentState:
    """Consulta el inventario actual con información del repuesto y localización."""
    try:
        async with get_connection() as conn:
            cursor = await conn.execute("""
                SELECT
                    r.referencia,
                    r.nombre AS nombre_repuesto,
                    r.marca,
                    r.tipo,
                    i.cantidad,
                    i.cantidad_minima,
                    i.posicion,
                    l.nombre AS localizacion,
                    CASE
                        WHEN i.cantidad = 0 THEN 'sin stock'
                        WHEN i.cantidad <= i.cantidad_minima THEN 'stock bajo'
                        ELSE 'normal'
                    END AS estado_stock
                FROM inventario i
                JOIN repuestos r ON i.id_repuesto = r.id_repuesto
                JOIN localizacion l ON i.id_localizacion = l.id_localizacion
                ORDER BY l.nombre, r.nombre
                LIMIT 300
            """)
            rows = await cursor.fetchall()

            columnas = [
                "referencia", "nombre_repuesto", "marca", "tipo",
                "cantidad", "cantidad_minima", "posicion",
                "localizacion", "estado_stock"
            ]
            datos = [dict(zip(columnas, row)) for row in rows]

        state.contexto_db.append({"fuente": "inventario", "datos": datos})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_inventario",
            "mensaje": f"Error consultando inventario: {str(e)}",
            "recuperable": True
        })

    return state


async def consultar_garantias(state: AgentState) -> AgentState:
    """Consulta garantías con repuesto, localización y usuario que reportó."""
    try:
        async with get_connection() as conn:
            cursor = await conn.execute("""
                SELECT
                    g.id_garantia::text,
                    g.referencia_repuesto,
                    g.nombre_repuesto,
                    g.estado,
                    g.motivo_falla,
                    g.comentarios_resolucion,
                    g.orden,
                    g.solicitante,
                    g.kilometraje,
                    l.nombre AS localizacion,
                    u.nombre AS usuario_reporta,
                    t.nombre AS tecnico_asociado,
                    g.created_at::text AS fecha_creacion,
                    g.updated_at::text AS fecha_actualizacion
                FROM garantias g
                JOIN localizacion l ON g.id_localizacion = l.id_localizacion
                JOIN usuarios u ON g.id_usuario_reporta = u.id_usuario
                LEFT JOIN usuarios t ON g.id_tecnico_asociado = t.id_usuario
                ORDER BY g.created_at DESC
                LIMIT 150
            """)
            rows = await cursor.fetchall()

            columnas = [
                "id_garantia", "referencia_repuesto", "nombre_repuesto",
                "estado", "motivo_falla", "comentarios_resolucion",
                "orden", "solicitante", "kilometraje", "localizacion",
                "usuario_reporta", "tecnico_asociado",
                "fecha_creacion", "fecha_actualizacion"
            ]
            datos = [dict(zip(columnas, row)) for row in rows]
            
            # DEBUG
            print(f"GARANTIAS - Registros encontrados: {len(datos)}")

        state.contexto_db.append({"fuente": "garantias", "datos": datos})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_garantias",
            "mensaje": f"Error consultando garantías: {str(e)}",
            "recuperable": True
        })

    return state


async def consultar_movimientos_tecnicos(state: AgentState) -> AgentState:
    """Consulta movimientos técnicos con repuesto, técnico y orden."""
    try:
        async with get_connection() as conn:
            cursor = await conn.execute("""
                SELECT
                    r.referencia,
                    r.nombre AS nombre_repuesto,
                    mt.concepto::text AS concepto,
                    mt.tipo::text AS tipo,
                    mt.cantidad,
                    mt.numero_orden,
                    mt.descargada,
                    l.nombre AS localizacion,
                    u.nombre AS responsable,
                    t.nombre AS tecnico_asignado,
                    mt.fecha::text AS fecha
                FROM movimientos_tecnicos mt
                JOIN repuestos r ON mt.id_repuesto = r.id_repuesto
                JOIN localizacion l ON mt.id_localizacion = l.id_localizacion
                JOIN usuarios u ON mt.id_usuario_responsable = u.id_usuario
                JOIN usuarios t ON mt.id_tecnico_asignado = t.id_usuario
                ORDER BY mt.fecha DESC
                LIMIT 150
            """)
            rows = await cursor.fetchall()

            columnas = [
                "referencia", "nombre_repuesto", "concepto", "tipo",
                "cantidad", "numero_orden", "descargada",
                "localizacion", "responsable", "tecnico_asignado", "fecha"
            ]
            datos = [dict(zip(columnas, row)) for row in rows]

        state.contexto_db.append({"fuente": "movimientos_tecnicos", "datos": datos})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_movimientos_tecnicos",
            "mensaje": f"Error consultando movimientos técnicos: {str(e)}",
            "recuperable": True
        })

    return state


async def consultar_solicitudes(state: AgentState) -> AgentState:
    """Consulta solicitudes con origen, destino, estado y trazabilidad."""
    try:
        async with get_connection() as conn:
            # Consulta principal de solicitudes
            cursor = await conn.execute("""
                SELECT
                    s.id_solicitud::text,
                    s.estado,
                    lo.nombre AS origen,
                    ld.nombre AS destino,
                    us.nombre AS solicitante,
                    ua.nombre AS alistador,
                    ur.nombre AS receptor,
                    s.fecha_creacion::text AS fecha_creacion,
                    s.fecha_alistamiento::text AS fecha_alistamiento,
                    s.fecha_despacho::text AS fecha_despacho,
                    s.fecha_recepcion::text AS fecha_recepcion,
                    s.guia_transporte,
                    s.observaciones_generales
                FROM solicitudes s
                JOIN localizacion lo ON s.id_localizacion_origen = lo.id_localizacion
                JOIN localizacion ld ON s.id_localizacion_destino = ld.id_localizacion
                JOIN usuarios us ON s.id_usuario_solicitante = us.id_usuario
                LEFT JOIN usuarios ua ON s.id_usuario_alistador = ua.id_usuario
                LEFT JOIN usuarios ur ON s.id_usuario_receptor = ur.id_usuario
                ORDER BY s.fecha_creacion DESC
                LIMIT 100
            """)
            rows = await cursor.fetchall()

            columnas = [
                "id_solicitud", "estado", "origen", "destino",
                "solicitante", "alistador", "receptor",
                "fecha_creacion", "fecha_alistamiento",
                "fecha_despacho", "fecha_recepcion",
                "guia_transporte", "observaciones_generales"
            ]
            datos = [dict(zip(columnas, row)) for row in rows]

        state.contexto_db.append({"fuente": "solicitudes", "datos": datos})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_solicitudes",
            "mensaje": f"Error consultando solicitudes: {str(e)}",
            "recuperable": True
        })

    return state


async def consultar_conteos(state: AgentState) -> AgentState:
    """Consulta conteos con sus detalles de diferencias por repuesto."""
    try:
        async with get_connection() as conn:
            # Conteos principales
            rows_conteos = await conn.execute("""
                SELECT
                    rc.id_conteo::text,
                    rc.tipo,
                    l.nombre AS localizacion,
                    u.nombre AS usuario,
                    rc.total_items_auditados,
                    rc.total_diferencia_encontrada,
                    rc.total_items_pq,
                    rc.observaciones,
                    rc.created_at::text AS fecha
                FROM registro_conteo rc
                JOIN localizacion l ON rc.id_localizacion = l.id_localizacion
                JOIN usuarios u ON rc.id_usuario = u.id_usuario
                ORDER BY rc.created_at DESC
                LIMIT 50
            """)
            rows = await cursor.fetchall()

            columnas_conteos = [
                "id_conteo", "tipo", "localizacion", "usuario",
                "total_items_auditados", "total_diferencia_encontrada",
                "total_items_pq", "observaciones", "fecha"
            ]
            datos_conteos = [dict(zip(columnas_conteos, row)) for row in rows_conteos]

            # Detalles de los conteos (diferencias por repuesto)
            rows_detalles = await conn.execute("""
                SELECT
                    dc.id_conteo::text,
                    r.referencia,
                    r.nombre AS nombre_repuesto,
                    dc.cantidad_sistema,
                    dc.cantidad_csa,
                    dc.diferencia,
                    dc.cantidad_pq
                FROM detalles_conteo dc
                JOIN repuestos r ON dc.id_repuesto = r.id_repuesto
                WHERE dc.diferencia != 0
                ORDER BY ABS(dc.diferencia) DESC
                LIMIT 100
            """)
            rows = await cursor.fetchall()

            columnas_detalles = [
                "id_conteo", "referencia", "nombre_repuesto",
                "cantidad_sistema", "cantidad_csa", "diferencia", "cantidad_pq"
            ]
            datos_detalles = [dict(zip(columnas_detalles, row)) for row in rows_detalles]

        state.contexto_db.append({"fuente": "conteos", "datos": datos_conteos})
        state.contexto_db.append({"fuente": "detalles_conteos", "datos": datos_detalles})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_conteos",
            "mensaje": f"Error consultando conteos: {str(e)}",
            "recuperable": True
        })

    return state


async def consultar_repuestos(state: AgentState) -> AgentState:
    """Consulta el catálogo de repuestos con toda su información."""
    try:
        async with get_connection() as conn:
            cursor = await conn.execute("""
                SELECT
                    referencia,
                    nombre,
                    marca,
                    tipo,
                    descripcion,
                    descontinuado,
                    fecha_estimada::text AS fecha_estimada,
                    created_at::text AS fecha_creacion
                FROM repuestos
                ORDER BY nombre
                LIMIT 300
            """)
            rows = await cursor.fetchall()

            columnas = [
                "referencia", "nombre", "marca", "tipo",
                "descripcion", "descontinuado",
                "fecha_estimada", "fecha_creacion"
            ]
            datos = [dict(zip(columnas, row)) for row in rows]

        state.contexto_db.append({"fuente": "repuestos", "datos": datos})

    except Exception as e:
        state.errores.append({
            "nodo": "consulta_repuestos",
            "mensaje": f"Error consultando repuestos: {str(e)}",
            "recuperable": True
        })

    return state


# --- Mapa de despacho ---
# El ejecutor usa este diccionario para encontrar qué función
# correr según la intención que detectó el clasificador.
HERRAMIENTAS: dict[str, callable] = {
    "inventario": consultar_inventario,
    "garantias": consultar_garantias,
    "movimientos_tecnicos": consultar_movimientos_tecnicos,
    "solicitudes": consultar_solicitudes,
    "conteos": consultar_conteos,
    "repuestos": consultar_repuestos,
}