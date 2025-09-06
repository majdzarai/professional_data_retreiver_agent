# report_generator/llm_client.py
import requests
import logging
import json

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        """
        Client for interacting with local Ollama models.
        Args:
            model (str): Name of the Ollama model to use.
            host (str): Base URL of the Ollama server.
        """
        self.model = model
        self.ollama_url = f"{host}/api/generate"
        logger.info(f"LLMClient initialized with model {self.model} at {self.ollama_url}")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text from the model using Ollama.
        Args:
            prompt (str): The input prompt.
            max_tokens (int): Max number of tokens to generate.
        Returns:
            str: Generated text from the model.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=30.0)
            response.raise_for_status()

            output = ""
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    try:
                        chunk = json.loads(data).get("response", "")
                        output += chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON chunk: {data}")
                        continue

            # Wrap the output in a JSON array format for the planner
            if output.strip() and not output.strip().startswith("["):
                # If output is not already in JSON format, wrap it in a JSON structure
                try:
                    # Try to see if it's already valid JSON
                    json.loads(output.strip())
                except json.JSONDecodeError:
                    # If not valid JSON, wrap it in our expected format
                    output = json.dumps([{
                        "question": output.strip(),
                        "queries": [output.strip()]
                    }])
            
            return output.strip()

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise ConnectionError(f"Failed to connect to Ollama API at {self.ollama_url}. Please ensure Ollama is running with the {self.model} model loaded.") from e
