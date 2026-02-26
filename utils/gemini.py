"""Cliente LLM híbrido con fallback automático y selección inteligente de modelo.

ARQUITECTURA:
1. Groq (primario) - Rápido, 14,400/día gratis
   - Clasificación: llama-3.1-8b-instant (ultra rápido)
   - Respuestas: llama-3.3-70b-versatile (mejor calidad)
2. Gemini (secundario) - Buena calidad, 30/día
   - Usa gemini-2.0-flash-exp para todo

El sistema intenta usar cada modelo en orden. Si uno falla por límite,
automáticamente prueba el siguiente.
"""

import os
from typing import Optional, List, Dict
from enum import Enum


class Provider(str, Enum):
    """Proveedores de LLM disponibles."""
    GROQ = "groq"
    GEMINI = "gemini"


class LLMClient:
    """Cliente LLM con fallback automático y selección inteligente de modelo."""

    def __init__(self):
        self.providers = self._init_providers()
        
        if not self.providers:
            raise ValueError(
                "No hay proveedores de LLM configurados. "
                "Necesitas al menos GROQ_API_KEY o GEMINI_API_KEY"
            )
        
        print(f"[LLM] Proveedores configurados: {[p['name'] for p in self.providers]}")
        print(f"[LLM] Prioridad: {self.providers[0]['name']}")

    def _init_providers(self) -> List[Dict]:
        """Inicializa los proveedores disponibles en orden de prioridad."""
        providers = []
        
        # 1. GROQ (prioridad 1 - más rápido y generoso)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                providers.append({
                    "name": Provider.GROQ,
                    "client": Groq(api_key=groq_key),
                    "model_fast": "llama-3.1-8b-instant",      # Para clasificación
                    "model_quality": "llama-3.3-70b-versatile", # Para respuestas
                    "active": True
                })
                print("[LLM] ✓ Groq configurado (fast: 8b-instant, quality: 70b-versatile)")
            except ImportError:
                print("[LLM] ✗ Groq no disponible (instala: pip install groq)")
        
        # 2. GEMINI (prioridad 2 - buena calidad)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                providers.append({
                    "name": Provider.GEMINI,
                    "client": genai.Client(api_key=gemini_key),
                    "model_fast": "gemini-2.0-flash-exp",    # Usa el mismo para ambos
                    "model_quality": "gemini-2.0-flash-exp",
                    "active": True
                })
                print("[LLM] ✓ Gemini configurado")
            except ImportError:
                print("[LLM] ✗ Gemini no disponible (instala: pip install google-genai)")
        
        return providers

    def _llamar_groq(
        self,
        provider: Dict,
        prompt: str,
        system_prompt: Optional[str],
        temperatura: float,
        max_tokens: int,
        use_quality_model: bool = False
    ) -> str:
        """Llama a Groq con selección de modelo según la tarea."""
        # Seleccionar modelo según el caso de uso
        model = provider["model_quality"] if use_quality_model else provider["model_fast"]
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        response = provider["client"].chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperatura,
            max_tokens=max_tokens,
        )

        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("Groq retornó una respuesta vacía")

        return response.choices[0].message.content.strip()

    def _llamar_gemini(
        self,
        provider: Dict,
        prompt: str,
        system_prompt: Optional[str],
        temperatura: float,
        max_tokens: int,
        use_quality_model: bool = False  # Ignorado, Gemini usa el mismo modelo
    ) -> str:
        """Llama a Gemini."""
        from google.genai import types
        
        model = provider["model_quality"] if use_quality_model else provider["model_fast"]
        
        contenido = []
        
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

        respuesta = provider["client"].models.generate_content(
            model=model,
            contents=contenido,
            config=types.GenerateContentConfig(
                temperature=temperatura,
                max_output_tokens=max_tokens
            )
        )

        if not respuesta.text or respuesta.text.strip() == "":
            raise RuntimeError("Gemini retornó una respuesta vacía")

        return respuesta.text.strip()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detecta si el error es por límite de cuota/rate."""
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            "429", "quota", "rate_limit", "resource_exhausted",
            "too many requests", "limit exceeded"
        ])

    def llamar(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperatura: float = 0.3,
        max_tokens: int = 2048,
        use_quality_model: bool = False  # NUEVO: selección inteligente de modelo
    ) -> str:
        """Llama al LLM con fallback automático entre proveedores.
        
        Args:
            prompt: El mensaje del usuario
            system_prompt: Instrucciones del sistema
            temperatura: Control de creatividad (0.0 - 1.0)
            max_tokens: Límite de tokens en la respuesta
            use_quality_model: Si True, usa modelo de mejor calidad (para respuestas finales).
                              Si False, usa modelo rápido (para clasificación).
        """
        
        last_error = None
        
        # Intentar cada proveedor activo en orden
        for i, provider in enumerate(self.providers):
            if not provider["active"]:
                continue
            
            try:
                # Llamar según el tipo de proveedor
                if provider["name"] == Provider.GROQ:
                    return self._llamar_groq(
                        provider, prompt, system_prompt, 
                        temperatura, max_tokens, use_quality_model
                    )
                
                elif provider["name"] == Provider.GEMINI:
                    return self._llamar_gemini(
                        provider, prompt, system_prompt,
                        temperatura, max_tokens, use_quality_model
                    )
                
            except Exception as e:
                last_error = e
                
                # Si es error de rate limit, marcar como inactivo temporalmente
                if self._is_rate_limit_error(e):
                    provider["active"] = False
                    print(f"[LLM] {provider['name']} alcanzó su límite, intentando siguiente proveedor...")
                    
                    # Si hay más proveedores, continuar
                    if i < len(self.providers) - 1:
                        continue
                else:
                    # Otro tipo de error, lanzar inmediatamente
                    raise RuntimeError(f"Error en {provider['name']}: {str(e)}") from e
        
        # Si llegamos aquí, todos los proveedores fallaron
        if last_error and self._is_rate_limit_error(last_error):
            raise RuntimeError(
                "Todos los proveedores de LLM alcanzaron sus límites. "
                "El servicio estará disponible nuevamente en unas horas. "
                "Considera agregar más API keys o upgrade a plan pagado."
            )
        else:
            raise RuntimeError(f"Error en todos los proveedores. Último error: {str(last_error)}")


# Instancia única - nombre 'gemini' para compatibilidad
gemini = LLMClient()