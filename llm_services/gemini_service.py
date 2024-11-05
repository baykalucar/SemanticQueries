from pydantic import BaseModel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from utils.environment_utils import gemini_settings_from_dot_env  # Replace with your env configuration
import google.generativeai as genai

# Define the CompletionResult class
class CompletionResult:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __iter__(self):
        return iter([self.content])  # Ensures CompletionResult can be iterated over

# Create the GeminiChatCompletion class
class GeminiChatCompletion(ChatCompletionClientBase, BaseModel):
    service_id: str
    model: str = "gemini-1.5-flash"  # Default model name
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True

    async def complete_async(self, messages: list, **kwargs):
        if self.debug:
            print("Gemini API Request:", messages)
        
        api_key = gemini_settings_from_dot_env()
        # Initialize the Gemini API client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        
        # Generate the content using the Gemini model
        prompt = " ".join([msg["content"] for msg in messages])
        response = model.generate_content(prompt)
        
        # Extract the text response
        content = response.text if response else ""
        
        if self.debug:
            print("Gemini API Response:", content)
        
        # Yield the single completion result
        return CompletionResult(content=content, metadata={"model": self.model})

    async def complete_chat(self, chat_history, settings, **kwargs):
        messages = [{"role": "user", "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages)
        return [result]  # Return a list of CompletionResult objects

    async def complete_chat_stream(self, chat_history, settings, **kwargs):
        messages = [{"role": "user", "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages)
        
        # Mock streaming by yielding each result in a single chunk
        yield result

def add_gemini_service(kernel, debug=False):
# Define the Gemini API settings (model)
    gemini_model = "gemini-1.5-flash"  # Adjust as needed
    service_id = "gemini_chat_completion"
    
    kernel.add_service(
        GeminiChatCompletion(service_id=service_id, model=gemini_model, debug=debug, ai_model_id=gemini_model),
    )