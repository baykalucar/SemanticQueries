import anthropic
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from pydantic import BaseModel  
from utils.environment_utils import claude_settings_from_dot_env  
# Define the CompletionResult class
class CompletionResult:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __iter__(self):
        return iter([self.content])  # Ensures CompletionResult can be iterated over

# Create the ClaudeChatCompletion class
class ClaudeChatCompletion(ChatCompletionClientBase, BaseModel):
    service_id: str
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"  # Example model
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True



    async def complete_async(self, messages: list, max_tokens: int = 500, temperature: float = 0.0, **kwargs):
        # Initialize the Anthropics API client
        client = anthropic.Client(api_key=self.api_key)
        
        # Prepare the payload for the request
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an expert software assistant specializing in generating SQL and Python code. When given a natural language query, respond with optimized, well-structured, and efficient code only. Provide explanations only if explicitly requested, focusing primarily on generating accurate SQL and Python snippets that solve the query effectively.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"]
                        }
                    ]
                } for msg in messages
            ]
        )

        # Check if response has content and that it's in the expected format
        if hasattr(response, "content") and isinstance(response.content, list):
            # Extract the text from each TextBlock in the content list
            content = " ".join([block.text for block in response.content if hasattr(block, "text")])
        else:
            raise ValueError("Unexpected response format")
        
        # Yield the single completion result
        return CompletionResult(content=content, metadata={"model": self.model})

    async def complete_chat(self, chat_history, settings, **kwargs):
        max_tokens = getattr(settings, 'max_tokens', 500)
        temperature = getattr(settings, 'temperature', 0.0)
        messages = [{"role": "user", "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages, max_tokens=max_tokens, temperature=temperature)
        return [result]  # Return a list of CompletionResult objects

    async def complete_chat_stream(self, chat_history, settings, **kwargs):
        max_tokens = getattr(settings, 'max_tokens', 500)
        temperature = getattr(settings, 'temperature', 0.0)
        messages = [{"role": "user", "content": msg.content} for msg in chat_history]
        
        # Await the result from complete_async
        result = await self.complete_async(messages, max_tokens=max_tokens, temperature=temperature)
        
        # Mock streaming by yielding each result in a single chunk
        yield result

def add_claude_service(kernel, debug=False):
     # Define the Claude API settings (model and token)
    claude_api_key, claude_model = claude_settings_from_dot_env()
    service_id = "claude_chat_completion"
    kernel.add_service(
        ClaudeChatCompletion(service_id=service_id, api_key=claude_api_key, ai_model_id=claude_model, model=claude_model, debug=debug),
    )