from dotenv import dotenv_values
def claude_settings_from_dot_env():
    """
    Reads the Claude API settings from the .env file.

    Returns:
        tuple: A tuple containing the model ID, API token, and API URL.

    """
    config = dotenv_values(".env")
    model_id = config.get("CLAUDE_MODEL_ID", None)
    api_token = config.get("CLAUDE_API_KEY", None)
    return api_token, model_id


def hugging_face_settings_from_dot_env(p_model="Llama318BInstruct") :

    config = dotenv_values(".env")
    model_name = config.get("HUGGINGFACE_MODEL_NAME_" + p_model, None)
    api_key = config.get("HUGGINGFACE_API_KEY", None)
    api_url = config.get("HUGGINGFACE_API_URL_" + p_model, None)
    return model_name, api_key, api_url

def gemini_settings_from_dot_env():
    """
    Reads the Gemini API settings from the .env file.

    Returns:
        str: The API key for the Gemini API.

    """
    config = dotenv_values(".env")
    api_key = config.get("GEMINI_API_KEY", None)
    return api_key

def deepseek_settings_from_dot_env():
    """
    Reads the DeepSeek API settings from the .env file.

    Returns:
        str: The API key for the DeepSeek API.

    """
    config = dotenv_values(".env")
    api_key = config.get("DEEPSEEK_API_KEY", None)
    api_url = config.get("DEEPSEEK_API_URL", None) 
    model_name = config.get("DEEPSEEK_MODEL_NAME", None)

    return api_key, api_url, model_name

def llama_settings_from_dot_env():
    """
    Reads the Llama API settings from the .env file.

    Returns:
        str: The API key for the Llama API.

    """
    config = dotenv_values(".env")
    api_key = config.get("LLAMA_API_KEY", None)
    api_url = config.get("LLAMA_API_URL", None)
    model_name = config.get("LLAMA_MODEL_NAME", None)
    return api_key, api_url, model_name