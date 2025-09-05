#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Client for the Report Generator

This module provides a client for interacting with different LLM providers,
including local Llama models.
"""

import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLM providers."""

    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        """Initialize the LLM client.

        Args:
            model: The model to use (e.g., "gpt-4", "llama-3.1", "llama3.1")
            temperature: The temperature to use for generation
        """
        self.model = model
        self.temperature = temperature
        self.is_local_llama = "llama" in model.lower()
        self.use_ollama = True  # Use Ollama API by default for local models
        self.ollama_url = "http://localhost:11434/api/generate"
        self.local_llama_path = os.environ.get("LLAMA_PATH", "llama-cpp-python")
        
        logger.info(f"Initialized LLM client with model: {model}")

    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: The maximum number of tokens to generate

        Returns:
            The generated text
        """
        if self.is_local_llama:
            return self._generate_with_local_llama(prompt, max_tokens)
        else:
            # For now, just return a placeholder response
            # In a real implementation, this would call the OpenAI API or similar
            logger.warning("Using placeholder LLM response. Implement actual API calls.")
            return f"Placeholder response for prompt: {prompt[:50]}..."

    def _generate_with_local_llama(self, prompt: str, max_tokens: int) -> str:
        """Generate text using a local Llama model.

        Args:
            prompt: The prompt to send to the model
            max_tokens: The maximum number of tokens to generate

        Returns:
            The generated text
        """
        if self.use_ollama:
            return self._generate_with_ollama(prompt, max_tokens)
        else:
            return self._generate_with_llama_cpp(prompt, max_tokens)
    
    def _generate_with_ollama(self, prompt: str, max_tokens: int) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: The prompt to send to the model
            max_tokens: The maximum number of tokens to generate

        Returns:
            The generated text
        """
        try:
            import requests
            
            # Convert model name if needed (llama-3.1 -> llama3.1)
            model_name = self.model.lower()
            if model_name == "llama-3.1" or model_name == "llama-3.1-8b":
                model_name = "llama3.1"
            
            # Prepare the request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Make the request to Ollama API
            logger.info(f"Sending request to Ollama API for model: {model_name}")
            response = requests.post(self.ollama_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error generating response: {error_msg}"
                
        except Exception as e:
            logger.error(f"Unexpected error with Ollama API: {e}")
            return f"Error generating response: {e}"
    
    def _generate_with_llama_cpp(self, prompt: str, max_tokens: int) -> str:
        """Generate text using llama-cpp-python.

        Args:
            prompt: The prompt to send to the model
            max_tokens: The maximum number of tokens to generate

        Returns:
            The generated text
        """
        try:
            import tempfile
            
            # Save prompt to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
                temp_prompt_file = f.name
                f.write(prompt)

            # Prepare the command to run the local Llama model
            cmd = [
                "python", "-c",
                "import sys; from llama_cpp import Llama; "
                f"model = Llama(model_path='{self._get_llama_model_path()}', n_ctx=4096); "
                f"with open('{temp_prompt_file}', 'r', encoding='utf-8') as f: prompt = f.read(); "
                f"output = model.create_completion(prompt, max_tokens={max_tokens}, temperature={self.temperature}); "
                "print(output['choices'][0]['text'])"
            ]

            # Run the command and capture the output
            logger.info(f"Running local Llama model with command: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Clean up the temporary file
            if os.path.exists(temp_prompt_file):
                os.remove(temp_prompt_file)

            return result.stdout.strip()
            
        except ImportError:
            logger.error("llama-cpp-python is not installed. Please install it with: pip install llama-cpp-python")
            return "Error: llama-cpp-python is not installed. Please install it with: pip install llama-cpp-python"

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running local Llama model: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return f"Error generating response: {e}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error generating response: {e}"

    def _get_llama_model_path(self) -> str:
        """Get the path to the Llama model file.

        Returns:
            The path to the model file
        """
        # Check if the model path is specified in an environment variable
        model_path = os.environ.get("LLAMA_MODEL_PATH")
        if model_path and os.path.exists(model_path):
            return model_path
        
        # Default paths to check for Llama 3.1 models
        default_paths = [
            "./models/llama-3.1-8b.gguf",
            "./models/llama-3.1-8b-instruct.gguf",
            "./llama-3.1-8b.gguf",
            "./llama-3.1-8b-instruct.gguf",
            "../models/llama-3.1-8b.gguf",
            "../models/llama-3.1-8b-instruct.gguf",
            "~/models/llama-3.1-8b.gguf",
            "~/models/llama-3.1-8b-instruct.gguf",
            "C:/models/llama-3.1-8b.gguf",
            "C:/models/llama-3.1-8b-instruct.gguf",
            "C:/Users/majdz/models/llama-3.1-8b.gguf",
            "C:/Users/majdz/models/llama-3.1-8b-instruct.gguf",
        ]
        
        for path in default_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                logger.info(f"Found Llama model at: {expanded_path}")
                return expanded_path
        
        # If no model file is found, raise an error
        raise FileNotFoundError(
            "Could not find Llama model file. "
            "Please set the LLAMA_MODEL_PATH environment variable to the path of the model file, "
            "or place the model in one of the default locations."
        )