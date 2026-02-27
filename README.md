# Minca Agent - Asistente de IA para Minca Electric

Minca Agent es un servicio de agente conversacional basado en IA que permite a los usuarios consultar información sobre inventario industrial de repuestos eléctricos mediante lenguaje natural. El agente funciona como backend de una aplicación de voz (Dynamo) que procesa preguntas y retorna respuestas optimizadas para audio.

## Características Principales

- **Procesamiento de lenguaje natural**: El agente entiende preguntas en español y determina qué información necesita consultar
- **Consultas a múltiples categorías**: Inventario, garantías, movimientos técnicos, solicitudes entre bodegas, conteos y catálogo de repuestos
- **Arquitectura basada en grafos**: Utiliza LangGraph para orquestar el flujo de procesamiento de manera declarativa
- **Memoria conversacional**: Mantiene contexto entre preguntas de la misma sesión
- **Fallback automático**: Cambia entre proveedores de LLM (Groq/Gemini) si uno falla
- **Optimizado para voz**: Las respuestas se generan en formato natural, escritas en palabras y sin listas

## Arquitectura del Sistema

```
                    ┌─────────────────┐
                    │  FastAPI Server │
                    │   (main.py)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Estado Inicial │
                    │  (pregunta +    │
                    │   memoria)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  CLASIFICADOR   │
                    │  (classifier)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐     │    ┌────────▼────────┐
     │   ERROR FATAL   │     │    │  EJECUTOR DE    │
     │ (si falla LLM)  │     │    │   CONSULTAS      │
     └────────┬────────┘     │    │ (db_tools)       │
              │              │    └────────┬────────┘
              └──────────────┼─────────────┘
                             │
                    ┌────────▼────────┐
                    │ GENERADOR DE    │
                    │ RESPUESTA       │
                    │ (response_gen)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Respuesta     │
                    │   + Memoria    │
                    └─────────────────┘
```

### Componentes del Grafo (LangGraph)

1. **Clasificador** (`app/classifier.py`): Analiza la pregunta del usuario usando un LLM para detectar:
   - **Intenciones**: Qué categorías de datos necesita (inventario, garantías, etc.)
   - **Tipo de operación**: Lectura, inserción, actualización o eliminación

2. **Ejecutor de Consultas** (`app/query_executor.py`): Ejecuta consultas SQL en paralelo según las intenciones detectadas

3. **Generador de Respuesta** (`app/response_generator.py`): Toma los datos de la base de datos y genera una respuesta natural en español

### Flujo de Procesamiento

1. El cliente envía una pregunta con un `session_id` opcional
2. El clasificador determina qué información necesita el usuario
3. Si hay un error fatal en clasificación, se salta a generar respuesta de error
4. El ejecutor consulta la base de datos para cada intención válida
5. El generador crea una respuesta humanizada con los datos obtenidos
6. Se actualiza la memoria de la sesión con la pregunta y respuesta

## Estructura del Proyecto

```
minca_agent/
├── main.py                    # Servidor FastAPI y endpoints
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Configuración de contenedor
├── .env.example              # Variables de entorno de ejemplo
│
├── app/                      # Núcleo del agente LangGraph
│   ├── __init__.py
│   ├── graph.py              # Definición del grafo de LangGraph
│   ├── state.py              # Esquema de estado que fluye por el grafo
│   ├── classifier.py         # Nodo clasificador de intenciones
│   ├── query_executor.py     # Nodo ejecutor de consultas a DB
│   └── response_generator.py # Nodo generador de respuestas
│
├── utils/                    # Utilidades y clientes externos
│   ├── __init__.py
│   ├── gemini.py             # Cliente LLM con fallback (Groq/Gemini)
│   └── database.py           # Pool de conexiones PostgreSQL
│
└── tools/                    # Herramientas de consulta
    ├── __init__.py
    └── db_tools.py           # Funciones SQL para cada categoría
```

## Requisitos

- Python 3.12+
- PostgreSQL (Supabase)
- API Keys de Groq y/o Gemini

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repositorio>
   cd minca_agent
   ```

2. **Crear entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**:
   ```bash
   cp .env.example .env
   # Editar .env con tus credenciales
   ```

## Variables de Entorno

| Variable | Descripción |
|----------|-------------|
| `SUPABASE_DB_URL` | URL de conexión a PostgreSQL de Supabase |
| `GROQ_API_KEY` | API key de Groq (opcional, pero recomendado) |
| `GEMINI_API_KEY` | API key de Gemini (fallback) |
| `AGENT_SERVICE_SECRET` | Secret compartido con Edge Functions |
| `PORT` | Puerto del servidor (default: 8000) |

## Uso Local

### Iniciar el servidor:
```bash
uvicorn main:app --reload
```

### Endpoint de salud:
```bash
GET http://localhost:8000/health
```

### Procesar una pregunta:
```bash
POST http://localhost:8000/procesar-pregunta
Authorization: Bearer <AGENT_SERVICE_SECRET>
Content-Type: application/json

{
  "pregunta": "¿Cuántos filtros hay en la bodega?",
  "session_id": null  // Opcional: tu propio session_id
}
```

## Despliegue

### Docker

```bash
docker build -t minca-agent .
docker run -p 8000:8000 --env-file .env minca-agent
```

### Railway / Render

El proyecto está configurado para desplegarse en Railway o Render:
- Puerto: 8000
- Health check: `/health`
- Requiere las variables de entorno configuradas

## Categorías Soportadas

| Categoría | Descripción |
|-----------|-------------|
| `inventario` | Stock de repuestos por localización |
| `garantías` | Estado de garantías, motivos de falla |
| `movimientos_tecnicos` | Movimientos de repuestos por técnicos |
| `solicitudes` | Solicitudes entre bodegas |
| `conteos` | Auditorías físicas y diferencias |
| `repuestos` | Catálogo de repuestos |

## Seguridad

- **Sin SQL dinámico**: Todas las consultas están escritas hardcodeadas
- **Autenticación**: El endpoint requiere un Bearer token compartido
- **Pool de conexiones**: Reutiliza conexiones a la base de datos

## Licencia

MIT
