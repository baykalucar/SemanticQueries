# Core modules
import semantic_kernel as sk
import datetime
import os
import json
import sqlite3
import pandas as pd
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union

# Enums and constants
from enum import Enum
from services import Service

# Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "database.db")
PLUGINS_DIRECTORY = "plugins"
DATA_SCHEMA_PATH = "data_schema.txt"


# Interfaces
class IKernelFactory(ABC):
    @abstractmethod
    def create_kernel(self, service: Service, model_name: str, model_mode: str, debug: bool) -> Any:
        """Create and configure a semantic kernel."""
        pass


class IDataRepository(ABC):
    @abstractmethod
    def execute_query(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute an SQL query and return the results."""
        pass
    
    @abstractmethod
    def save_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Save a DataFrame to CSV."""
        pass


class IFileRepository(ABC):
    @abstractmethod
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        pass
    
    @abstractmethod
    def write_file(self, content: str, file_path: str) -> None:
        """Write content to a file."""
        pass
    
    @abstractmethod
    def ensure_directory_exists(self, directory_path: str) -> None:
        """Ensure a directory exists, create if it doesn't."""
        pass


class ITextParser(ABC):
    @abstractmethod
    def parse_text_between_tags(self, text: str, start_tag: str, end_tag: str) -> List[str]:
        """Parse text between specified tags."""
        pass
    
    @abstractmethod
    def parse_sql(self, result_string: str) -> Optional[str]:
        """Extract SQL query from text."""
        pass
    
    @abstractmethod
    def parse_python_code(self, result_string: str) -> Optional[str]:
        """Extract Python code from text."""
        pass
    
    @abstractmethod
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
        pass


class IPromptProcessor(ABC):
    @abstractmethod
    async def process_prompt(self, data_schema: str, prompt: str, **kwargs) -> str:
        """Process a prompt through a model."""
        pass
    
    @abstractmethod
    async def rephrase_prompt(self, data_schema: str, prompt: str) -> str:
        """Rephrase a user prompt to be more effective for the model."""
        pass


# Implementations
class KernelFactory(IKernelFactory):
    def create_kernel(self, service: Service, model_name: str = "Llama318BInstruct", 
                     model_mode: str = "chat", debug: bool = False) -> Any:
        """Create a semantic kernel with configured settings."""
        from llm_services.kernel_service import initialize_kernel
        return initialize_kernel(selected_service=service, model_name=model_name, 
                                model_mode=model_mode, debug=debug)


class SQLiteRepository(IDataRepository):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def execute_query(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame."""
        try:
            conn = sqlite3.connect(self.connection_string)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"SQL execution error: {e}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to CSV file."""
        try:
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            raise


class FileRepository(IFileRepository):
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            if not os.path.exists(file_path):
                return ""
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def write_file(self, content: str, file_path: str) -> None:
        """Write content to a file."""
        try:
            self.ensure_directory_exists(os.path.dirname(file_path))
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")
            raise
    
    def ensure_directory_exists(self, directory_path: str) -> None:
        """Ensure a directory exists, create if it doesn't."""
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)


class TextParser(ITextParser):
    def parse_text_between_tags(self, text: str, start_tag: str, end_tag: str) -> List[str]:
        """Extract text between specific tags."""
        from utils.parse_utils import parse_text_between_tags
        return parse_text_between_tags(text, start_tag, end_tag)
    
    def parse_sql(self, result_string: str) -> Optional[str]:
        """Extract SQL query from result string."""
        matches = self.parse_text_between_tags(result_string, "<sql>", "</sql>")
        if not matches:
            matches = self.parse_text_between_tags(result_string, "<SQL>", "</SQL>")
        if not matches:
            matches = self.parse_text_between_tags(result_string, "```sql", "```")
        
        if matches:
            return matches[0].replace("\\_", "_").replace("[", "").replace("]", "")
        return None
    
    def parse_python_code(self, result_string: str) -> Optional[str]:
        """Extract Python code from result string."""
        matches = self.parse_text_between_tags(result_string, "<python>", "</python>")
        if not matches:
            matches = self.parse_text_between_tags(result_string, "<PYTHON>", "</PYTHON>")
        if not matches:
            matches = self.parse_text_between_tags(result_string, "```python", "```")
        
        if matches:
            return matches[0].replace("\\_", "_").replace("> ", "")
        return None
    
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Extract first valid JSON object from text."""
        try:
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"sql_score": 0, "python_code": 0}
        except Exception as e:
            print(f"Failed to extract JSON: {e}")
            return {"sql_score": 0, "python_score": 0}


class PromptProcessor(IPromptProcessor):
    def __init__(self, kernel_factory: IKernelFactory, service: Service = Service.AzureOpenAI,
                model_name: str = "Llama318BInstruct", model_mode: str = "chat", 
                debug: bool = False):
        self.kernel_factory = kernel_factory
        self.service = service
        self.model_name = model_name
        self.model_mode = model_mode
        self.debug = debug
        self.kernel = self.kernel_factory.create_kernel(service, model_name, model_mode, debug)
    
    async def process_prompt(self, data_schema: str, prompt: str, 
                           plugin_name: str = "DataPlugin", function_name: str = "DatabaseDescriptor",
                           **kwargs) -> str:
        """Process a prompt using the semantic kernel."""
        plugin_functions = self.kernel.import_plugin_from_prompt_directory(PLUGINS_DIRECTORY, plugin_name)
        target_function = plugin_functions[function_name]
        
        arguments = sk.KernelArguments(
            data_schema=data_schema,
            user_prompt=prompt,
            **kwargs
        )
        
        result = await self.kernel.invoke(target_function, arguments)
        
        # Handle different result formats
        if hasattr(result, 'data'):
            return result.data
        elif result.__dict__:
            completion_results = result.value
            if isinstance(completion_results, list) and completion_results:
                return completion_results[0].content
        
        return str(result)
    
    async def rephrase_prompt(self, data_schema: str, prompt: str) -> str:
        """Rephrase a user prompt to be more effective."""
        return await self.process_prompt(
            data_schema=data_schema,
            prompt=prompt,
            plugin_name="PromptPlugin",
            function_name="PromptRephraser",
            query=prompt
        )


class PythonCodeExecutor:
    def __init__(self, db_connection: str, debug: bool = False):
        self.db_connection = db_connection
        self.debug = debug
    
    def execute(self, code: str, dataframe: Optional[pd.DataFrame] = None) -> None:
        """Execute Python code with access to the database and dataframe."""
        if self.debug:
            print(f"PYTHON:\n{code}")
        
        try:
            # Create an environment with necessary variables
            local_vars = {'conn': sqlite3.connect(self.db_connection), 'df': dataframe}
            
            # Execute the code with the prepared environment
            exec(code, globals(), local_vars)
            
            # Close connection
            if 'conn' in local_vars and local_vars['conn']:
                local_vars['conn'].close()
                
        except Exception as e:
            print(f"Python execution error: {e}")
            raise


class QueryProcessor:
    def __init__(self, 
                 kernel_factory: IKernelFactory,
                 data_repository: IDataRepository,
                 file_repository: IFileRepository,
                 text_parser: ITextParser,
                 code_executor: PythonCodeExecutor,
                 service: Service = Service.AzureOpenAI,
                 model_name: str = "Llama318BInstruct",
                 model_mode: str = "chat",
                 debug: bool = False):
        self.kernel_factory = kernel_factory
        self.data_repository = data_repository
        self.file_repository = file_repository
        self.text_parser = text_parser
        self.code_executor = code_executor
        self.service = service
        self.model_name = model_name
        self.model_mode = model_mode
        self.debug = debug
        self.prompt_processor = PromptProcessor(
            kernel_factory, service, model_name, model_mode, debug
        )
    
    async def process_query(self, 
                          user_prompt: Optional[str] = None,
                          prompt_rephrase: bool = False,
                          output_dir: str = "") -> Optional[pd.DataFrame]:
        """Process a user query to generate SQL, execute it, and optionally visualize results."""
        # Get user input if not provided
        if user_prompt is None:
            user_prompt = input("Enter your query: ")
        
        # Load data schema
        data_schema = self.file_repository.read_file(DATA_SCHEMA_PATH)
        
        # Rephrase prompt if requested
        prompt = user_prompt
        if prompt_rephrase:
            if self.debug:
                print("Rephrasing prompt...")
            prompt = await self.prompt_processor.rephrase_prompt(data_schema, user_prompt)
        
        # Save the user prompt
        if output_dir:
            self.file_repository.write_file(user_prompt, os.path.join(output_dir, "user_prompt.txt"))
        
        # Generate response with SQL and Python code
        if self.debug:
            print("Generating SQL and Python code with LLM...")
        
        save_plot_instruction = ""
        if output_dir:
            save_plot_instruction = f"Generated plots should be saved in the directory: {output_dir}plot.png"
        
        result = await self.prompt_processor.process_prompt(
            data_schema=data_schema,
            prompt=prompt,
            plugin_name="DataPlugin", 
            function_name="DatabaseDescriptor",
            save_plot_to_disk=save_plot_instruction
        )
        
        # Extract SQL and Python code
        sql = self.text_parser.parse_sql(result)
        python_code = self.text_parser.parse_python_code(result)
        
        # Save extracted code if output directory is provided
        if output_dir:
            if sql:
                self.file_repository.write_file(sql, os.path.join(output_dir, "sql_query.txt"))
            if python_code:
                self.file_repository.write_file(python_code, os.path.join(output_dir, "python_code.txt"))
        
        # Execute SQL if present
        dataframe = None
        if sql:
            print(f"User query: {user_prompt}")
            if prompt_rephrase:
                print(f"Rephrased prompt: {prompt}")
            
            if self.debug:
                print(f"SQL: {sql}")
            
            try:
                dataframe = self.data_repository.execute_query(sql)
            except Exception as e:
                if self.debug:
                    print(f"SQL execution error: {str(e)}")
                dataframe = await self.fix_sql(data_schema, prompt, sql, str(e), output_dir)
        
        # Execute Python code if present and we have data
        if python_code and dataframe is not None:
            try:
                self.code_executor.execute(python_code, dataframe)
            except Exception as e:
                if self.debug:
                    print(f"Python execution error: {str(e)}")
                await self.fix_python_code(data_schema, prompt, python_code, str(e), output_dir, dataframe)
        
        # Save results to CSV if requested
        if dataframe is not None and not dataframe.empty and output_dir:
            self.data_repository.save_to_csv(dataframe, os.path.join(output_dir, "output.csv"))
        
        return dataframe
    
    async def fix_sql(self, data_schema: str, prompt: str, sql: str, error: str, 
                     output_dir: str, iteration: int = 1) -> Optional[pd.DataFrame]:
        """Attempt to fix SQL query errors by requesting corrections from the model."""
        if iteration > 3:
            print("Could not fix the query after 3 attempts. Please check the query and try again.")
            return None
        
        if self.debug:
            print(f"SQL fix iteration {iteration}: Error: {error}")
        
        # Request fixed SQL
        fixed_result = await self.prompt_processor.process_prompt(
            data_schema=data_schema,
            prompt=prompt,
            plugin_name="DataPlugin",
            function_name="SQLFixer",
            sql_query=sql,
            error=error
        )
        
        fixed_sql = self.text_parser.parse_sql(fixed_result)
        
        if fixed_sql:
            if output_dir:
                self.file_repository.write_file(
                    fixed_sql, 
                    os.path.join(output_dir, f"fixed_sql_{iteration}.txt")
                )
            
            try:
                return self.data_repository.execute_query(fixed_sql)
            except Exception as e:
                return await self.fix_sql(
                    data_schema, prompt, fixed_sql, str(e), output_dir, iteration + 1
                )
        
        return None
    
    async def fix_python_code(self, data_schema: str, prompt: str, python_code: str, 
                            error: str, output_dir: str, dataframe: pd.DataFrame, 
                            iteration: int = 1) -> None:
        """Attempt to fix Python code errors by requesting corrections from the model."""
        if iteration > 3:
            print("Could not fix the Python code after 3 attempts. Please check the code and try again.")
            return
        
        if self.debug:
            print(f"Python fix iteration {iteration}: Error: {error}")
        
        # Request fixed Python code
        fixed_result = await self.prompt_processor.process_prompt(
            data_schema=data_schema,
            prompt=prompt,
            plugin_name="DataPlugin",
            function_name="PythonFixer",
            python_code=python_code,
            error=error
        )
        
        fixed_code = self.text_parser.parse_python_code(fixed_result)
        
        if fixed_code:
            if output_dir:
                self.file_repository.write_file(
                    fixed_code, 
                    os.path.join(output_dir, f"fixed_python_{iteration}.txt")
                )
            
            try:
                self.code_executor.execute(fixed_code, dataframe)
            except Exception as e:
                await self.fix_python_code(
                    data_schema, prompt, fixed_code, str(e), output_dir, dataframe, iteration + 1
                )


class QuestionGenerator:
    def __init__(self, 
                 kernel_factory: IKernelFactory,
                 file_repository: IFileRepository,
                 service: Service = Service.AzureOpenAI,
                 model_name: str = "Llama318BInstruct",
                 model_mode: str = "chat",
                 debug: bool = False):
        self.kernel_factory = kernel_factory
        self.file_repository = file_repository
        self.service = service
        self.model_name = model_name
        self.model_mode = model_mode
        self.debug = debug
        self.prompt_processor = PromptProcessor(
            kernel_factory, service, model_name, model_mode, debug
        )
    
    async def generate_questions(self) -> str:
        """Generate sample questions based on the data schema."""
        data_schema = self.file_repository.read_file(DATA_SCHEMA_PATH)
        
        print("Generating questions...")
        result = await self.prompt_processor.process_prompt(
            data_schema=data_schema,
            prompt="",
            plugin_name="DataPlugin",
            function_name="QuestionGenerator"
        )
        
        # Save questions to file
        filename = os.path.join(
            "questions", 
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}.txt"
        )
        self.file_repository.ensure_directory_exists(os.path.dirname(filename))
        self.file_repository.write_file(result, filename)
        
        print(f"Result saved to: {filename}")
        print(result)
        
        return result


class ScoreGenerator:
    def __init__(self, 
                 kernel_factory: IKernelFactory,
                 file_repository: IFileRepository,
                 text_parser: ITextParser,
                 service: Service = Service.AzureOpenAI,
                 model_name: str = "Llama318BInstruct",
                 model_mode: str = "chat",
                 debug: bool = False):
        self.kernel_factory = kernel_factory
        self.file_repository = file_repository
        self.text_parser = text_parser
        self.service = service
        self.model_name = model_name
        self.model_mode = model_mode
        self.debug = debug
        self.prompt_processor = PromptProcessor(
            kernel_factory, service, model_name, model_mode, debug
        )
    
    async def generate_scores(self, folder: str, question: str, sql_query: str, 
                            python_code: str) -> Dict[str, float]:
        """Generate evaluation scores for SQL and Python code."""
        data_schema = self.file_repository.read_file(DATA_SCHEMA_PATH)
        
        print("Generating scores...")
        result = await self.prompt_processor.process_prompt(
            data_schema=data_schema,
            prompt="",
            plugin_name="DataPlugin",
            function_name="ScoreGenerator",
            question=question,
            sql_query=sql_query,
            python_code=python_code
        )
        
        # Save scores to file
        scores_dir = os.path.join(folder, "scores")
        self.file_repository.ensure_directory_exists(scores_dir)
        
        filename = os.path.join(
            scores_dir, 
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}.txt"
        )
        self.file_repository.write_file(result, filename)
        
        print(f"Result saved to: {filename}")
        print(result)
        
        # Extract scores from result
        return self.text_parser.extract_json(result)


class AnswerGenerator:
    def __init__(self, 
                 query_processor: QueryProcessor,
                 file_repository: IFileRepository,
                 service: Service = Service.AzureOpenAI,
                 debug: bool = False):
        self.query_processor = query_processor
        self.file_repository = file_repository
        self.service = service
        self.debug = debug
    
    async def process_questions_file(self, filename: str) -> None:
        """Read questions from a JSON file and generate answers for each."""
        content = self.file_repository.read_file(filename)
        
        # Extract base filename without extension for directory structure
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        directory_path = os.path.join("answers", self.service.value, base_filename)
        self.file_repository.ensure_directory_exists(directory_path)
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {filename}: {e}")
            return
        
        for i, item in enumerate(data):
            complexity = item.get("complexity", "unknown")
            queries = item.get("queries", [])
            
            for question in queries:
                print(f"Index: {i}")
                print(f"Question: {question}")
                print(f"Complexity: {complexity}")
                
                question_folder = os.path.join(directory_path, f"{i}_{complexity}")
                self.file_repository.ensure_directory_exists(question_folder)
                
                try:
                    await self.query_processor.process_query(
                        user_prompt=question,
                        prompt_rephrase=False,
                        output_dir=f"{question_folder}/"
                    )
                except Exception as e:
                    error_message = str(e)
                    self.file_repository.write_file(
                        error_message, 
                        os.path.join(question_folder, "error.txt")
                    )
                    if self.debug:
                        print(f"Error processing question: {error_message}")


class ScoreProcessor:
    def __init__(self, 
                 score_generator: ScoreGenerator,
                 file_repository: IFileRepository):
        self.score_generator = score_generator
        self.file_repository = file_repository
    
    async def process_scores(self, question_date: str, output_excel_path: str) -> None:
        """Process and aggregate scores for all models and questions."""
        base_folder = "answers"
        results = []
        
        for model in Service:
            model_folder = os.path.join(base_folder, model.value, question_date)
            if not os.path.exists(model_folder):
                print(f"Skipping {model_folder}, folder does not exist.")
                continue
            
            # Process each subfolder (question)
            for subfolder_name in os.listdir(model_folder):
                subfolder_path = os.path.join(model_folder, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue
                
                try:
                    index, category = subfolder_name.split('_', 1)
                except ValueError:
                    print(f"Skipping {subfolder_name}, unexpected folder name format.")
                    continue
                
                # Read files
                sql_query = self.file_repository.read_file(
                    os.path.join(subfolder_path, "sql_query.txt")
                )
                python_code = self.file_repository.read_file(
                    os.path.join(subfolder_path, "python_code.txt")
                )
                question = self.file_repository.read_file(
                    os.path.join(subfolder_path, "user_prompt.txt")
                )
                
                # Generate scores
                scores = await self.score_generator.generate_scores(
                    folder=subfolder_path,
                    question=question,
                    sql_query=sql_query,
                    python_code=python_code
                )
                
                sql_score = scores.get("sql_score", 0)
                python_score = scores.get("python_score", 0)
                avg_score = (sql_score + python_score) / 2
                
                # Collect result
                results.append({
                    "Answer Model": model.value,
                    "Question Category": category,
                    "Question #": index,
                    "SQL (Data Result)": sql_score,
                    "Python (Visualization)": python_score,
                    "Avg Score": avg_score,
                    "FileName": question_date
                })
        
        # Create DataFrame and save to Excel
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_excel_path, index=False)
            print(f"Results saved to {output_excel_path}")
        else:
            print("No results were generated.")


# Facade for main functionality
class QueryFacade:
    def __init__(self, service: Service = Service.AzureOpenAI, 
                model_name: str = "Llama318BInstruct", 
                model_mode: str = "chat",
                debug: bool = False):
        # Initialize dependencies
        self.kernel_factory = KernelFactory()
        self.file_repository = FileRepository()
        self.text_parser = TextParser()
        self.data_repository = SQLiteRepository(DB_CONNECTION_STRING)
        self.code_executor = PythonCodeExecutor(DB_CONNECTION_STRING, debug)
        
        # Initialize processors
        self.query_processor = QueryProcessor(
            self.kernel_factory,
            self.data_repository,
            self.file_repository,
            self.text_parser,
            self.code_executor,
            service,
            model_name,
            model_mode,
            debug
        )
        
        self.question_generator = QuestionGenerator(
            self.kernel_factory,
            self.file_repository,
            service,
            model_name,
            model_mode,
            debug
        )
        
        self.score_generator = ScoreGenerator(
            self.kernel_factory,
            self.file_repository,
            self.text_parser,
            service,
            model_name,
            model_mode,
            debug
        )
        
        self.answer_generator = AnswerGenerator(
            self.query_processor,
            self.file_repository,
            service,
            debug
        )
        
        self.score_processor = ScoreProcessor(
            self.score_generator,
            self.file_repository
        )
    
    async def prompt_to_query_result(self, debug: bool = False, 
                                  prompt_rephrase: bool = False, 
                                  user_prompt: Optional[str] = None, 
                                  output_dir: str = "") -> Optional[pd.DataFrame]:
        """Process a user prompt to SQL query and visualization."""
        return await self.query_processor.process_query(
            user_prompt=user_prompt,
            prompt_rephrase=prompt_rephrase,
            output_dir=output_dir
        )
    
    async def generate_questions(self) -> str:
        """Generate sample questions based on the data schema."""
        return await self.question_generator.generate_questions()
    
    async def read_questions_and_generate_answers(self, filename: str) -> None:
        """Process a file of questions and generate answers."""
        await self.answer_generator.process_questions_file(filename)
    
    async def process_scores(self, question_date: str, output_excel_path: str) -> None:
        """Process and aggregate scores from answers."""
        await self.score_processor.process_scores(question_date, output_excel_path)


# Example usage
async def main():
    # Initialize facade with desired configuration
    facade = QueryFacade(
        service=Service.AzureOpenAI,
        model_name="",
        model_mode="",
        debug=True
    )
    
    # Example: Process a user query
    df = await facade.prompt_to_query_result()



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())