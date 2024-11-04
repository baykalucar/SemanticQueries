import semantic_kernel as sk
import datetime
import os
import json
import sqlite3
from llm_services.kernel_service import initialize_kernel
from utils.file_utils import  read_data_schema_from_file, write_to_file
from utils.sql_lit_db_utils import run_sql_query
from utils.parse_utils import parse_text_between_tags
from services import Service

async def PromptToQueryResult(debug=False, prompt_rephrase=False, selected_service=Service.AzureOpenAI, query=None, outputFileDir="", model_name="Llama318BInstruct", model_mode="chat"):
    """
    Prompts the user for a query, rephrases the prompt if required, and executes the query using the Semantic Kernel.

    Args:
        debug (bool, optional): If True, prints debug information. Defaults to False.
        prompt_rephrase (bool, optional): If True, rephrases the prompt using a rephraser plugin. Defaults to False.

    Returns:
        DataFrame or any: The result of the executed query, or 'any' if no query is executed.
    """
    kernel = initialize_kernel(selected_service = selected_service, model_name=model_name, model_mode=model_mode, debug= debug)

    
    if(query is None):
        query = input("Enter your query: ")  # Get query from user
    rephrased_prompt = query
    
    plugins_directory = "plugins"
    file_path = "data_schema.txt"
    data_schema = read_data_schema_from_file(file_path)
    if(prompt_rephrase):
        promptFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, "PromptPlugin")
        rephraserFunction = promptFunctions["PromptRephraser"]
        rephrased_prompt_result = await kernel.invoke(rephraserFunction, sk.KernelArguments(data_schema=data_schema, query=query))
        # Debug: Print the type and value of the result to understand its structure
        # print(type(rephrased_prompt_result))  # This will show whether it's a tuple, list, or other type
        # print(rephrased_prompt_result)        # This will print the raw result
        # print("Kernel invoke result:", rephrased_prompt_result.__dict__)
        if(hasattr(rephrased_prompt_result, 'data')):
            rephrased_prompt = rephrased_prompt_result.data
        elif (rephrased_prompt_result.__dict__):
            # Access the value (which contains the list of CompletionResult objects)
            # print("Value: ", rephrased_prompt_result.value)
            completion_results = rephrased_prompt_result.value

            # Check if it's a list and access the content of the first CompletionResult
            if isinstance(completion_results, list) and len(completion_results) > 0:
                first_result = completion_results[0]
                # print(first_result.content)  # This will print the content of the first result
                rephrased_prompt = first_result.content
            else:
                print("No completion results found.")
        else:
            rephrased_prompt = str(rephrased_prompt_result)
        #rephrased_prompt = rephrased_prompt_result.data if hasattr(rephrased_prompt_result, 'data') else str(rephrased_prompt_result)

    dataFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, "DataPlugin")
    descriptorFunction = dataFunctions["DatabaseDescriptor"]

    savePlotToDisk = ""
    if(outputFileDir != ""):
        savePlotToDisk = "Generated plots should be saved in the directory: " + outputFileDir + "plot.png"

    if(outputFileDir != ""):
        # Write rephrased prompt to query.txt file
        with open(outputFileDir + "user_prompt.txt", "w") as file:
            file.write(rephrased_prompt)

    result = await kernel.invoke(descriptorFunction, sk.KernelArguments(data_schema=data_schema, query= rephrased_prompt, save_plot_to_disk = savePlotToDisk))
    if(hasattr(result, 'data')):
        result_string = result.data
    elif (result.__dict__):
        # Access the value (which contains the list of CompletionResult objects)
        # print("Value: ", result.value)
        completion_results = result.value

        # Check if it's a list and access the content of the first CompletionResult
        if isinstance(completion_results, list) and len(completion_results) > 0:
            first_result = completion_results[0]
            # print(first_result.content)  # This will print the content of the first result
            result_string = first_result.content
        else:
            print("No completion results found.")
    else:
        result_string = str(result)
    if(debug):
        print("result String:", result_string)
    matches_sql = parse_text_between_tags(result_string,"<sql>", "</sql>")
    if(len(matches_sql) == 0):
        matches_sql = parse_text_between_tags(result_string,"<SQL>", "</SQL>")

    if(prompt_rephrase):
        print("User query: " + query)
    print("Rephrased prompt: " + rephrased_prompt + "#")
    if len(matches_sql) > 0:
        sql = matches_sql[0]
        if(outputFileDir != ""):
            # Write query to .txt file
            write_to_file(sql, outputFileDir + "sql_query.txt")
        if debug:
            print("SQL: ", sql)
        df = run_sql_query(sql)

    matches_python = parse_text_between_tags(result_string,"<python>", "</python>")
    if(len(matches_python) == 0):
        matches_python = parse_text_between_tags(result_string,"<PYTHON>", "</PYTHON>")
    if len(matches_python) > 0:
        if debug:
            print("PYTHON:", matches_python[0])
        try:
            db_conn = os.getenv("DB_CONNECTION_STRING")
            conn = sqlite3.connect(db_conn)
            exec(matches_python[0].replace("\\_", "_"))
            conn.close()

            if(outputFileDir != ""):
                # Write python code to .txt file
                with open(outputFileDir + "python_code.txt", "w") as file:
                    file.write(matches_python[0])

        except Exception as e:
            print('hata:' + e.__str__() )
        except:
            print("An exception occurred")
    if len(matches_sql) > 0:
        df.head()
        if(outputFileDir != ""):
            df.to_csv(outputFileDir + "output.csv", index=False)
        return df
    else:  
        return any

async def GenerateQuestions(selected_service=Service.AzureOpenAI,huggingface_model="Llama318BInstruct"):
    """
    Generates possible questions using the Semantic Kernel and prints the results.
    """
    kernel = initialize_kernel(selected_service, huggingface_model)
    
    plugins_directory = "plugins"
    file_path = "data_schema.txt"
    data_schema = read_data_schema_from_file(file_path)
    promptFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, "DataPlugin")
    queryGeneratorFunction = promptFunctions["QuestionGenerator"]
    result = await kernel.invoke(queryGeneratorFunction, sk.KernelArguments(data_schema=data_schema))
    if(hasattr(result, 'data')):
        result_string = result.data
    elif (result.__dict__):
        # Access the value (which contains the list of CompletionResult objects)
        # print("Value: ", result.value)
        completion_results = result.value

        # Check if it's a list and access the content of the first CompletionResult
        if isinstance(completion_results, list) and len(completion_results) > 0:
            first_result = completion_results[0]
            # print(first_result.content)  # This will print the content of the first result
            result_string = first_result.content
        else:
            print("No completion results found.")
    else:
        result_string = str(result)
    # Generate a unique filename based on the current date
    filename = "questions/" + datetime.datetime.now().strftime("%Y-%m-%d") + ".txt"

    # Write result_string to the file
    write_to_file(result_string, filename)

    # Print the filename
    print("Result saved to:", filename)
    print(result_string)

async def ReadQuestionsAndGenerateAnswers(filename, debug=False, selectedService=Service.AzureOpenAI):
    """
    Reads questions from a file and generates answers using the Semantic Kernel.

    Args:
        filename (str): The name of the file containing the questions.

    """
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the filename from the full path
    filename = os.path.basename(filename)

    # Remove the file extension
    filename = os.path.splitext(filename)[0]

    # Create the directory path
    directory_path = "answers/" + filename 

    print("Directory path: ", directory_path)   

    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    

    data = json.loads(content)
    i = 0
    for item in data:
        complexity = item["complexity"]
        queries = item["queries"]
        
        for question in queries:
            print("Index:" , i)
            print("Question: ", question)
            print("Complexity: ", complexity)
            questionFolderName = i.__str__() + "_" + complexity
            if not os.path.exists(directory_path + "/"  + questionFolderName + "/"):
                os.makedirs(directory_path + "/"  + questionFolderName + "/")

            try:
                await PromptToQueryResult(debug=debug, prompt_rephrase=False, selectedService=selectedService, query=question, outputFileDir=directory_path + "/" + questionFolderName + "/")
            except Exception as e:
                error_message = str(e)
                with open(directory_path + "/" + questionFolderName + "/error.txt", "w") as file:
                    file.write(error_message)
            i = i + 1