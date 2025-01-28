from pydantic import BaseModel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from utils.environment_utils import deepseek_settings_from_dot_env  # Replace with your env configuration
import httpx

# Define the CompletionResult class
class CompletionResult:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __iter__(self):
        return iter([self.content])  # Ensures CompletionResult can be iterated over

# Create the DeepSeekChatCompletion class
class DeepSeekChatCompletion(ChatCompletionClientBase, BaseModel):
    service_id: str
    model: str = "deepseek-chat"  # Default model name
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def complete_async(self, messages: list, **kwargs):
        # Load API key from environment
        api_key, api_url, model_name = deepseek_settings_from_dot_env()

        self.model = model_name

        
        # Prepare the payload for the API request
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": msg["content"]
                } for msg in messages
            ],
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Define a custom timeout (e.g., 30 seconds)
        timeout = httpx.Timeout(60.0, connect=15.0) # 60 seconds for the request, 10 seconds to connect
        
        # Send the request to the DeepSeek API
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{api_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()  # Raise an error for HTTP codes >= 400
                data = response.json()
        except httpx.ReadTimeout:
            raise RuntimeError("The request to DeepSeek API timed out. Please try again or increase the timeout.")
        except httpx.RequestError as e:
            raise RuntimeError(f"An error occurred while requesting DeepSeek API: {e}")
        
        # Extract the generated content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Yield the single completion result
        return CompletionResult(content=content, metadata={"model": self.model})

    async def complete_chat(self, chat_history, settings, **kwargs):
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages)
        return [result]  # Return a list of CompletionResult objects

    async def complete_chat_stream(self, chat_history, settings, **kwargs):
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages)
        
        # Mock streaming by yielding each result in a single chunk
        yield result

def add_deepseek_service(kernel, debug=False):
    # Define the DeepSeek API settings (model)
    deepseek_model = "deepseek-chat"  # Adjust as needed
    service_id = "deepseek_chat_completion"
    
    kernel.add_service(
        DeepSeekChatCompletion(service_id=service_id, model=deepseek_model, debug=debug, ai_model_id=deepseek_model),
    )
