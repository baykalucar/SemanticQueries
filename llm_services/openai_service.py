from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import semantic_kernel as sk
def add_openai_service(kernel):
    api_key, org_id = sk.openai_settings_from_dot_env()
    service_id = "gpt4-32k"
    kernel.add_service(
        OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt4-32k", api_key=api_key, org_id=org_id),
    )