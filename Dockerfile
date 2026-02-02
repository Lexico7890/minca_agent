# Imagen base ligera de Python
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo requirements primero para aprovechar el cache de Docker.
# Si no cambian las dependencias, Docker no las reinstala en cada build.
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Puerto que expone el contenedor.
# Railway lo detecta automáticamente si usas esta variable.
ENV PORT=8000

# Comando que inicia el servidor.
# --host 0.0.0.0 permite que sea accesible desde fuera del contenedor.
# --port $PORT usa la variable de entorno PORT.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port $PORT"