"""Cliente LLM híbrido con fallback automático y métricas de uso.

CAMBIOS:
- Gemini usa modelo ESTABLE (no experimental)
- Contador de requests para debugging
- Logs claros de consumo
"""

import os
from typing import Optional, List, Dict
from enum import Enum


class Provider(str, Enum):
    """Proveedores de LLM disponibles."""
    GROQ = "groq"
    GEMINI = "gemini"


class LLMClient:
    """Cliente LLM con fallback automático y monitoreo de uso."""

    def __init__(self):
        self.providers = self._init_providers()
        # Contadores de requests para debugging
        self.request_counts = {}
        
        if not self.providers:
            raise ValueError(
                "No hay proveedores de LLM configurados. "
                "Necesitas al menos GROQ_API_KEY o GEMINI_API_KEY"
            )
        
        # Inicializar contadores
        for p in self.providers:
            self.request_counts[p['name']] = 0
        
        print(f"[LLM] Proveedores configurados: {[p['name'] for p in self.providers]}")
        print(f"[LLM] Prioridad: {self.providers[0]['name']}")

    def _init_providers(self) -> List[Dict]:
        """Inicializa los proveedores disponibles."""
        providers = []
        
        # 1. GROQ
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                providers.append({
                    "name": Provider.GROQ,
                    "client": Groq(api_key=groq_key),
                    "model_fast": "llama-3.1-8b-instant",
                    "model_quality": "llama-3.3-70b-versatile",
                    "active": True
                })
                print("[LLM] ✓ Groq configurado (fast: 8b-instant, quality: 70b-versatile)")
            except Exception as e:
                print(f"[LLM] ✗ Error con Groq: {e}")
        
        # 2. GEMINI - MODELO ESTABLE
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                providers.append({
                    "name": Provider.GEMINI,
                    "client": genai.Client(api_key=gemini_key),
                    # CAMBIADO: usar modelo estable, no experimental
                    "model_fast": "gemini-1.5-flash",
                    "model_quality": "gemini-1.5-flash",
                    "active": True
                })
                print("[LLM] ✓ Gemini configurado (modelo: gemini-1.5-flash)")
            except Exception as e:
                print(f"[LLM] ✗ Error con Gemini: {e}")
        
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
        """Llama a Groq."""
        model = provider["model_quality"] if use_quality_model else provider["model_fast"]
        
        print(f"[Groq] Usando modelo: {model}")
        
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

        if not response.choices:
            raise RuntimeError("Groq: respuesta vacía")

        return response.choices[0].message.content.strip()

    def _llamar_gemini(
        self,
        provider: Dict,
        prompt: str,
        system_prompt: Optional[str],
        temperatura: float,
        max_tokens: int,
        use_quality_model: bool = False
    ) -> str:
        """Llama a Gemini."""
        from google.genai import types
        
        model = provider["model_quality"] if use_quality_model else provider["model_fast"]
        
        print(f"[Gemini] Usando modelo: {model}")
        
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

        if not respuesta.text:
            raise RuntimeError("Gemini: respuesta vacía")

        return respuesta.text.strip()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detecta errores de límite."""
        error_msg = str(error).lower()
        return any(kw in error_msg for kw in [
            "429", "quota", "rate_limit", "resource_exhausted", "too many"
        ])

    def llamar(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperatura: float = 0.3,
        max_tokens: int = 2048,
        use_quality_model: bool = False
    ) -> str:
        """Llama al LLM con fallback automático."""
        
        last_error = None
        
        for i, provider in enumerate(self.providers):
            if not provider["active"]:
                continue
            
            try:
                # Incrementar contador
                self.request_counts[provider['name']] += 1
                
                print(f"[LLM] Request #{self.request_counts[provider['name']]} a {provider['name']}")
                
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
                print(f"[LLM] Error en {provider['name']}: {str(e)[:200]}")
                
                if self._is_rate_limit_error(e):
                    provider["active"] = False
                    print(f"[LLM] {provider['name']} límite alcanzado (requests usados: {self.request_counts[provider['name']]})")
                    print(f"[LLM] Intentando siguiente proveedor...")
                    if i < len(self.providers) - 1:
                        continue
                else:
                    raise RuntimeError(f"Error en {provider['name']}: {str(e)}") from e
        
        if last_error and self._is_rate_limit_error(last_error):
            raise RuntimeError("Todos los proveedores alcanzaron sus límites")
        else:
            raise RuntimeError(f"Error en todos los proveedores: {str(last_error)}")


gemini = LLMClient()