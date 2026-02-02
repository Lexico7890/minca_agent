"""Conexión a PostgreSQL.

Maneja el pool de conexiones al servidor de Supabase.
El pool se inicializa una sola vez al iniciar el servicio
y se reutiliza para todas las consultas del agente.
"""

import os
from typing import Optional
import psycopg
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager

# Variable global del pool. Se setea en startup().
_pool: Optional[AsyncConnectionPool] = None


async def startup():
    """Inicializa el pool de conexiones.
    
    Se llama una sola vez cuando el servicio de FastAPI inicia.
    min_size=2 significa que siempre hay 2 conexiones abiertas listas.
    max_size=10 es el límite superior si hay muchas consultas simultáneas.
    """
    global _pool

    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL no está definida en las variables de entorno")

    _pool = AsyncConnectionPool(
        conninfo=db_url,
        min_size=2,
        max_size=10,
        open=False
    )
    await _pool.open()


async def shutdown():
    """Cierra el pool cuando el servicio se apaga."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection():
    """Context manager que entrega una conexión del pool.
    
    Uso:
        async with get_connection() as conn:
            rows = await conn.execute("SELECT ...").fetchall()
    
    La conexión se retorna automáticamente al pool al salir del bloque,
    sin importar si hubo error o no.
    """
    global _pool
    if _pool is None:
        raise RuntimeError("Pool de conexiones no inicializado")

    async with _pool.connection() as conn:
        # autocommit=True para queries de lectura (SELECT).
        # Cuando agregamos escritura (INSERT/UPDATE) lo manejamos
        # con transacciones explícitas en cada herramienta.
        await conn.set_autocommit(True)
        yield conn