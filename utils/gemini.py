"""Cliente de Gemini.

Centraliza toda la comunicación con la API de Google Gemini.
Todos los nodos del grafo usan esta clase en lugar de
instanciar el SDK directamente, así tenemos un solo lugar
para configurar el modelo, manejar errores de API y
eventualmente agregar reintentos.
"""

import os
from google import genai
from google.genai import types


class GeminiClient:
    """Wrapper del SDK de Gemini que expone un método simple para llamar al modelo.
    
    Se instancia una sola vez al iniciar el servicio y se reutiliza
    en todos los nodos del grafo.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está definida en las variables de entorno")

        # Cliente del SDK de Google
        self.client = genai.Client(api_key=api_key)

        # Modelo que usamos. gemini-2.0-flash es el más rápido
        # y tiene capa gratuita generosa.
        self.model = "gemini-2.0-flash"

    def llamar(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperatura: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Llama al modelo de Gemini y retorna la respuesta como texto.
        
        Args:
            prompt: El mensaje principal (la pregunta o instrucción)
            system_prompt: Instrucciones de sistema opcionales (rol del modelo, reglas)
            temperatura: Cuánto "creativo" es el modelo. 0.0 = determinista, 1.0 = más creativo.
                         Para clasificación usamos valor bajo, para respuestas más alto.
            max_tokens: Límite de tokens en la respuesta.
        
        Returns:
            El texto de la respuesta del modelo.
        
        Raises:
            RuntimeError: Si la API retorna un error o la respuesta está vacía.
        """
        try:
            contenido = []

            # Si hay system_prompt, lo agregamos como mensaje de sistema
            if system_prompt:
                contenido.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"[INSTRUCCIONES DEL SISTEMA]\n{system_prompt}\n\n[MENSAJE]\n{prompt}")]
                    )
                )
            else:
                contenido.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)]
                    )
                )

            respuesta = self.client.models.generate_content(
                model=self.model,
                contents=contenido,
                config=types.GenerateContentConfig(
                    temperature=temperatura,
                    max_output_tokens=max_tokens
                )
            )

            # Verificar que la respuesta tiene contenido
            if not respuesta.text or respuesta.text.strip() == "":
                raise RuntimeError("El modelo retornó una respuesta vacía")

            return respuesta.text.strip()

        except RuntimeError:
            raise  # Re-lanzar nuestros propios errores
        except Exception as e:
            raise RuntimeError(f"Error llamando a Gemini: {str(e)}") from e


# Instancia única que se usa en todo el proyecto.
# Se crea cuando el módulo se importa por primera vez.
gemini = GeminiClient()