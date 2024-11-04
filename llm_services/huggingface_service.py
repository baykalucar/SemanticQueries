from huggingface_hub import InferenceApi
from pydantic import BaseModel
import requests
import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from utils.environment_utils import hugging_face_settings_from_dot_env

# Define a simple completion object to hold both the response and metadata
class CompletionResult:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __iter__(self):
        return iter([self.content])  # Ensures CompletionResult can be iterated over

class HuggingFaceChatCompletion(ChatCompletionClientBase, BaseModel):
    service_id: str
    api_token: str
    api_url: str
    mode: str = "chat"
    debug: bool = False
    ai_model_name: str

    class Config:
        arbitrary_types_allowed = True

    async def complete_async(self, messages: list, max_tokens: int = 2000, stream: bool = False, **kwargs):
        # Create the headers and payload for the request
        if(self.debug):
            print("HuggingFace API Request:", messages)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Add a new item to messages with role and content
        
        messages.insert(0, {"role": "system", "content": "You are an expert software assistant specializing in generating SQL and Python code. When given a natural language query, respond with optimized, well-structured, and efficient code only. Provide explanations only if explicitly requested, focusing primarily on generating accurate SQL and Python snippets that solve the query effectively."})

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                if self.debug:
                    print("Adding assistant message to chat history")
                # Insert the new message after the current user message
                messages.insert(i + 1, {"role": "assistant", "content": "Ok, please continue"})

        if(self.mode == "chat"):
            payload = {
                "model": self.api_url.split('/')[-4],  # model_id
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream
            }
        elif(self.mode == "text"):
            payload = {
                "inputs": messages,  # Use 'inputs' field for text completion
                "parameters": {"max_new_tokens": max_tokens}
            }
        elif(self.mode == "chattext"):
            payload = {
                "inputs": messages[0]['content'],  # Use 'inputs' field for text completion
                "parameters": {"max_new_tokens": max_tokens}
            }
        if(self.debug):
            print("HuggingFace API Request:", payload)
        # Make the request to HuggingFace API
        response = requests.post(self.api_url, headers=headers, json=payload)
        response_json = response.json()

        # Log the response for debugging
        # print("HuggingFace API Response:", response_json)
        if(self.debug):
            print("HuggingFace API Response:", response_json)
        # Handle the response format
        if "choices" in response_json:
            
            content = response_json["choices"][0]["message"]["content"]
            # Return a list of CompletionResult objects, even if there's only one
            return [CompletionResult(content=content, metadata={"model": self.ai_model_id})]
            #return content
        else:
            raise ValueError(f"Unexpected response format: {response_json}")

    # Updated complete_chat to handle chat_history and settings
    async def complete_chat(self, chat_history, settings, **kwargs):
        messages = [{"role": "user", "content": msg.content} for msg in chat_history]
        return await self.complete_async(messages)

    # Implement required complete_chat_stream method
    async def complete_chat_stream(self, chat_history, settings, **kwargs):
        # For non-streaming models, we can return the entire result as one chunk
        result = await self.complete_chat(chat_history, settings)
        for chunk in [result]:  # Mock streaming by yielding the entire result as a single chunk
            yield chunk

def add_huggingface_service(kernel: sk.Kernel, huggingface_model: str, model_mode: str = "chat", debug: bool = False):
    model_id, api_token, api_url = hugging_face_settings_from_dot_env(huggingface_model)
    # api_token = "hf_..."  # Set your HuggingFace API token here
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Set the model ID you want to use from HuggingFace
    # api_url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
    # Initialize the HuggingFace Inference API without task override
    hf_service = InferenceApi(repo_id=model_id, token=api_token)
    service_id = "hf_chat_completion"
    kernel.add_service(
        HuggingFaceChatCompletion(service_id=service_id, api_token=api_token, api_url=api_url, ai_model_id=model_id, mode=model_mode, max_tokens=5000, debug=debug, ai_model_name=huggingface_model),
    )