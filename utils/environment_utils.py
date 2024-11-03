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