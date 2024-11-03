from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import semantic_kernel as sk
def add_azure_openai_service(kernel):
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    service_id = "aoai_chat_completion"
    kernel.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment, endpoint=endpoint, api_key=api_key),
    )