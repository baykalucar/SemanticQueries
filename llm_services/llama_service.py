from pydantic import BaseModel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from utils.environment_utils import llama_settings_from_dot_env  # Replace with your environment settings
from openai import OpenAI, OpenAIError
import httpx

# Define the CompletionResult class
class CompletionResult:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __iter__(self):
        return iter([self.content])  # Ensures CompletionResult can be iterated over

# Create the LlamaChatCompletion class
class LlamaChatCompletion(ChatCompletionClientBase, BaseModel):
    service_id: str
    model: str = "llama3.1-70b"  # Default model name
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def complete_async(self, messages: list, **kwargs):
        # Load API key from environment
        api_key, base_url, model_name = llama_settings_from_dot_env()
        
        self.model = model_name
        
        # Create the OpenAI client with Llama API settings
        client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            # Send the request to the Llama API
            response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            # Extract the generated content
            content = response.choices[0].message.content if response.choices else ""
            return CompletionResult(content=content, metadata={"model": self.model})

        except OpenAIError as e:
            raise RuntimeError(f"An error occurred while requesting Llama API: {e}")
        except httpx.ReadTimeout:
            raise RuntimeError("The request to Llama API timed out. Please try again or increase the timeout.")

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

def add_llama_service(kernel, debug=False):
    # Define the Llama API settings (model)
    llama_model = "llama3.1-70b"  # Adjust as needed
    service_id = "llama_chat_completion"
    
    kernel.add_service(
        LlamaChatCompletion(service_id=service_id, model=llama_model, debug=debug, ai_model_id=llama_model),
    )
