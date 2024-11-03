import semantic_kernel as sk
from .claude_service import add_claude_service
from .huggingface_service import add_huggingface_service
from .claude_service import add_claude_service
from .openai_service import add_openai_service
from .azure_openai_service import add_azure_openai_service
from services import Service


def initialize_kernel(selected_service, huggingface_model:str, model_mode:str):
    kernel = sk.Kernel()
    if selected_service == Service.OpenAI:
        add_openai_service(kernel)
    elif selected_service == Service.AzureOpenAI:
        add_azure_openai_service(kernel)
    elif selected_service == Service.HuggingFace:
        add_huggingface_service(kernel, huggingface_model=huggingface_model, model_mode=model_mode)
    elif selected_service == Service.ClaudeAI:
        add_claude_service(kernel)
    return kernel
    