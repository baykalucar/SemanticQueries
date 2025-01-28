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

async def PromptToQueryResult(debug=False, prompt_rephrase=False, selected_service=Service.AzureOpenAI, 
                              user_prompt=None, outputFileDir="", model_name="Llama318BInstruct", model_mode="chat",
                              plugin_name="DataPlugin", function_name="DatabaseDescriptor" ):
    """
    Prompts the user for a query, rephrases the prompt if required, and executes the query using the Semantic Kernel.

    Args:
        debug (bool, optional): If True, prints debug information. Defaults to False.
        prompt_rephrase (bool, optional): If True, rephrases the prompt using a rephraser plugin. Defaults to False.

    Returns:
        DataFrame or any: The result of the executed query, or 'any' if no query is executed.
    """
    kernel = initialize_kernel(selected_service = selected_service, model_name=model_name, model_mode=model_mode, debug= debug)

    if(user_prompt is None):
        user_prompt = input("Enter your query: ")  # Get query from user
    rephrased_prompt = user_prompt
    
    plugins_directory = "plugins"
    file_path = "data_schema.txt"
    data_schema = read_data_schema_from_file(file_path)
    if(prompt_rephrase):
        if debug:
            print("Rephrasing prompt...")
        rephrased_prompt = await rephrase_prompt(kernel, plugins_directory, data_schema, user_prompt)
    if debug:
        print("Generating SQL and Python code with LLM...")
    result_string = await execute_llm_prompt(kernel, plugins_directory, data_schema, rephrased_prompt, outputFileDir,plugin_name, function_name)
    
    sql = parse_sql(result_string)
    python_code = parse_python_code(result_string)

    if sql is not None:
        try:
            df = execute_sql(debug, prompt_rephrase, user_prompt, outputFileDir, rephrased_prompt, sql)
        except Exception as e:
            df = await execute_fixed_query(kernel, plugins_directory, data_schema, debug, prompt_rephrase, user_prompt, rephrased_prompt, outputFileDir, plugin_name, sql, e.__str__())
        except BaseException as e:
            df = await execute_fixed_query(kernel, plugins_directory, data_schema, debug, prompt_rephrase, user_prompt, rephrased_prompt, outputFileDir, plugin_name, sql, e.__str__())
    
    if python_code is not None:
        try:
            execute_python_code(debug, outputFileDir, python_code, df)
        except Exception as e:
            if debug:
                print("Trying to fix error executing python code. ", e)
            await execute_fixed_python_code(kernel, plugins_directory, data_schema, debug, rephrased_prompt, outputFileDir, plugin_name, python_code, e.__str__(), 1, df)
        except BaseException as e:
            if debug:
                print("Trying to fix base error executing python code: ", e)
            await execute_fixed_python_code(kernel, plugins_directory, data_schema, debug, rephrased_prompt, outputFileDir, plugin_name, python_code, e.__str__(), 1, df)  
    
    if sql is not None:
        if df is not None:
            if not df.empty:
                df.head()
                if(outputFileDir != ""):
                    df.to_csv(outputFileDir + "output.csv", index=False)
                return df
            else:
                print("No data found in the DataFrame.")
                return None
        else:
            print("No DataFrame found.")
            return None
    else:
        print("No SQL code found.")
        return None

async def execute_fixed_query(kernel, plugins_directory, data_schema, debug, prompt_rephrase, user_prompt, rephrased_prompt, outputFileDir, plugin_name, sql, error, iteration=1):
    if(iteration > 3):
        print("Could not fix the query. Please check the query and try again.")
        return None
    if debug:
        print("SQL fix Iteration: ", iteration+1, " Error: ", error)
    
        fixed_sql_result_string = await execute_llm_prompt(kernel=kernel, plugins_directory=plugins_directory,data_schema=data_schema, 
                                                                rephrased_prompt=rephrased_prompt, outputFileDir=outputFileDir, plugin_name=plugin_name, function_name="SQLFixer", sql_query=sql, error=error)
        fixed_sql = parse_sql(fixed_sql_result_string)
        if fixed_sql is not None:
            try:
                df = execute_sql(debug, prompt_rephrase, user_prompt, outputFileDir, rephrased_prompt, fixed_sql)
                return df
            except Exception as e:
                await execute_fixed_query(kernel, plugins_directory, data_schema, debug, prompt_rephrase, user_prompt, rephrased_prompt, outputFileDir, plugin_name, sql, e.__str__(), iteration+1)

async def execute_fixed_python_code(kernel, plugins_directory, data_schema, debug, rephrased_prompt, outputFileDir, plugin_name, python_code, error, iteration, df):
    if(iteration > 3):
        print("Could not fix the python code. Please check the code and try again.")
        return None
    if debug:
        print("Python fix Iteration: ", iteration+1, " Error: ", error)
    
    fixed_python_code_result_string = await execute_llm_prompt(kernel=kernel, plugins_directory=plugins_directory,data_schema=data_schema, 
                                                            rephrased_prompt=rephrased_prompt, outputFileDir=outputFileDir, plugin_name=plugin_name, function_name="PythonFixer", python_code=python_code, error=error)
    fixed_python_code = parse_python_code(fixed_python_code_result_string)
    if fixed_python_code is not None:
        try:
            execute_python_code(debug, outputFileDir, fixed_python_code, df)
            return
        except Exception as e:
            await execute_fixed_python_code(kernel, plugins_directory, data_schema, debug, rephrased_prompt, outputFileDir, plugin_name, python_code, e.__str__(), iteration+1, df)
            
def execute_python_code(debug, outputFileDir, python_code, df):
    if debug:
        print("PYTHON:", python_code)

    db_conn = os.getenv("DB_CONNECTION_STRING")
    conn = sqlite3.connect(db_conn)
    exec(python_code)
    conn.close()

    if(outputFileDir != ""):
        # Write python code to .txt file
        with open(outputFileDir + "python_code.txt", "w") as file:
            file.write(python_code)

    
def parse_python_code(result_string):
    matches_python = parse_text_between_tags(result_string,"<python>", "</python>")
    if(len(matches_python) == 0):
        matches_python = parse_text_between_tags(result_string,"<PYTHON>", "</PYTHON>")
    if(len(matches_python) == 0):
        matches_python = parse_text_between_tags(result_string,"```python", "```")
    if(len(matches_python) > 0):
        return matches_python[0].replace("\\_", "_").replace("> ", "")
    return None

def parse_sql(result_string):
    matches_sql = parse_text_between_tags(result_string,"<sql>", "</sql>")
    if(len(matches_sql) == 0):
        matches_sql = parse_text_between_tags(result_string,"<SQL>", "</SQL>")
    if(len(matches_sql) == 0):
        matches_sql = parse_text_between_tags(result_string,"```sql", "```")
    if(len(matches_sql) > 0):
        return matches_sql[0].replace("\\_", "_").replace("[", "").replace("]", "")
    return None

def execute_sql(debug, prompt_rephrase, query, outputFileDir, rephrased_prompt, sql): 
    print("User query: " + query)
    if(prompt_rephrase):
        print("Rephrased prompt: " + rephrased_prompt + "#")
    
    if(outputFileDir != ""):
        # Write query to .txt file
        write_to_file(sql, outputFileDir + "sql_query.txt")
    if debug:
        print("SQL: ", sql)
    df = run_sql_query(sql)

    return df
    
async def execute_llm_prompt(kernel, plugins_directory, data_schema, rephrased_prompt, outputFileDir, plugin_name, function_name, sql_query="", python_code="", error=""):
    dataFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, plugin_name)
    descriptorFunction = dataFunctions[function_name]

    savePlotToDisk = ""
    if outputFileDir != "":
        savePlotToDisk = "Generated plots should be saved in the directory: " + outputFileDir + "plot.png"

    if outputFileDir != "":
        # Write rephrased prompt to query.txt file
        with open(outputFileDir + "user_prompt.txt", "w") as file:
            file.write(rephrased_prompt)

    result = await kernel.invoke(descriptorFunction, sk.KernelArguments(data_schema=data_schema, user_prompt=rephrased_prompt, save_plot_to_disk=savePlotToDisk, sql=sql_query, python_code=python_code, error=error))
    
    if hasattr(result, 'data'):
        result_string = result.data
    elif result.__dict__:
        # Access the value (which contains the list of CompletionResult objects)
        completion_results = result.value

        # Check if it's a list and access the content of the first CompletionResult
        if isinstance(completion_results, list) and len(completion_results) > 0:
            first_result = completion_results[0]
            result_string = first_result.content
        else:
            print("No completion results found.")
            result_string = None
    else:
        result_string = str(result)
    
    return result_string    

async def rephrase_prompt(kernel, plugins_directory, data_schema, query):
    promptFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, "PromptPlugin")
    rephraserFunction = promptFunctions["PromptRephraser"]
    rephrased_prompt_result = await kernel.invoke(rephraserFunction, sk.KernelArguments(data_schema=data_schema, query=query))

    if hasattr(rephrased_prompt_result, 'data'):
        return rephrased_prompt_result.data
    elif rephrased_prompt_result.__dict__:
        completion_results = rephrased_prompt_result.value

        if isinstance(completion_results, list) and len(completion_results) > 0:
            first_result = completion_results[0]
            return first_result.content
        else:
            print("No completion results found.")
            return None
    else:
        return str(rephrased_prompt_result)

async def GenerateQuestions(selected_service=Service.AzureOpenAI,model_name="Llama318BInstruct", model_mode="chat", debug=False):
    """
    Generates possible questions using the Semantic Kernel and prints the results.
    """
    kernel = initialize_kernel(selected_service, model_name=model_name, model_mode=model_mode, debug=debug)
    
    plugins_directory = "plugins"
    file_path = "data_schema.txt"
    data_schema = read_data_schema_from_file(file_path)
    print("Generating questions...")
    promptFunctions = kernel.import_plugin_from_prompt_directory(plugins_directory, "DataPlugin")
    queryGeneratorFunction = promptFunctions["QuestionGenerator"]
    result = await kernel.invoke(queryGeneratorFunction, sk.KernelArguments(data_schema=data_schema))
    if(hasattr(result, 'data')):
        print("Parsing result data...")
        result_string = result.data
    elif (result.__dict__):
        # Access the value (which contains the list of CompletionResult objects)
        # print("Value: ", result.value)
        print("Parsing result value...")
        completion_results = result.value

        # Check if it's a list and access the content of the first CompletionResult
        if isinstance(completion_results, list) and len(completion_results) > 0:
            first_result = completion_results[0]
            # print(first_result.content)  # This will print the content of the first result
            print("Parsing result content...")
            result_string = first_result.content
        else:
            print("No completion results found.")
    else:
        print("Parsing result string...")
        result_string = str(result)
    # Generate a unique filename based on the current date
    filename = "questions/" + datetime.datetime.now().strftime("%Y-%m-%d") + ".txt"

    # Write result_string to the file
    write_to_file(result_string, filename)

    # Print the filename
    print("Result saved to:", filename)
    print(result_string)

async def ReadQuestionsAndGenerateAnswers(filename, debug=False, selected_service=Service.AzureOpenAI):
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
    directory_path = "answers/" + selected_service.value + "/" + filename 

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
                await PromptToQueryResult(debug=debug, prompt_rephrase=False, selected_service=selected_service, user_prompt=question, outputFileDir=directory_path + "/" + questionFolderName + "/")
            except Exception as e:
                error_message = str(e)
                with open(directory_path + "/" + questionFolderName + "/error.txt", "w") as file:
                    file.write(error_message)
            i = i + 1