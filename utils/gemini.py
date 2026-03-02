"""Cliente LLM híbrido - VERSIÓN FINAL OPTIMIZADA.

CAMBIOS:
- Gemini usa gemini-2.5-flash (modelo estable que funciona)
- Logs de consumo de tokens
- Rate limiting mejorado
"""

import os
from typing import Optional, List, Dict
from enum import Enum


class Provider(str, Enum):
    """Proveedores de LLM disponibles."""
    GROQ = "groq"
    GEMINI = "gemini"


class LLMClient:
    """Cliente LLM con fallback automático."""

    def __init__(self):
        self.providers = self._init_providers()
        self.request_counts = {}
        
        if not self.providers:
            raise ValueError("No hay proveedores configurados")
        
        for p in self.providers:
            self.request_counts[p['name']] = 0
        
        print(f"[LLM] Proveedores: {[p['name'] for p in self.providers]}")

    def _init_providers(self) -> List[Dict]:
        """Inicializa proveedores."""
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
                print("[LLM] ✓ Groq OK")
            except Exception as e:
                print(f"[LLM] ✗ Groq error: {e}")
        
        # 2. GEMINI - MODELO CORRECTO
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                providers.append({
                    "name": Provider.GEMINI,
                    "client": genai.Client(api_key=gemini_key),
                    # MODELO CORRECTO: gemini-2.5-flash
                    "model_fast": "gemini-2.5-flash",
                    "model_quality": "gemini-2.5-flash",
                    "active": True
                })
                print("[LLM] ✓ Gemini OK (modelo: gemini-2.5-flash)")
            except Exception as e:
                print(f"[LLM] ✗ Gemini error: {e}")
        
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
        
        # Estimar tokens de entrada (aproximado)
        input_tokens = (len(prompt) + len(system_prompt or "")) // 4
        print(f"[Groq] Modelo: {model}, Tokens entrada: ~{input_tokens}")
        
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
            raise RuntimeError("Groq: sin respuesta")

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
        
        input_tokens = (len(prompt) + len(system_prompt or "")) // 4
        print(f"[Gemini] Modelo: {model}, Tokens entrada: ~{input_tokens}")
        
        contenido = []
        
        if system_prompt:
            contenido.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"[INSTRUCCIONES]\n{system_prompt}\n\n[MENSAJE]\n{prompt}")]
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
            raise RuntimeError("Gemini: sin respuesta")

        return respuesta.text.strip()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detecta rate limit."""
        error_msg = str(error).lower()
        return any(kw in error_msg for kw in [
            "429", "quota", "rate_limit", "resource_exhausted", 
            "too many", "tokens per minute", "tpm"
        ])

    def llamar(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperatura: float = 0.3,
        max_tokens: int = 2048,
        use_quality_model: bool = False
    ) -> str:
        """Llama al LLM con fallback."""
        
        last_error = None
        
        for i, provider in enumerate(self.providers):
            if not provider["active"]:
                continue
            
            try:
                self.request_counts[provider['name']] += 1
                print(f"[LLM] Request #{self.request_counts[provider['name']]} → {provider['name']}")
                
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
                error_preview = str(e)[:150]
                print(f"[LLM] Error en {provider['name']}: {error_preview}")
                
                if self._is_rate_limit_error(e):
                    provider["active"] = False
                    print(f"[LLM] {provider['name']} límite alcanzado. Fallback activado.")
                    if i < len(self.providers) - 1:
                        continue
                else:
                    raise RuntimeError(f"Error en {provider['name']}: {str(e)}") from e
        
        if last_error and self._is_rate_limit_error(last_error):
            raise RuntimeError("Todos los proveedores alcanzaron límites")
        else:
            raise RuntimeError(f"Error: {str(last_error)}")


gemini = LLMClient()