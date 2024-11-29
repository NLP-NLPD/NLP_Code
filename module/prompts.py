import os
import json
from langchain.prompts import PromptTemplate

class Prompts:
    # /templates 폴더에서 프롬프트를 불러와 프롬프트 생성. PromptTemplate 로 만든다.
    def __init__(self, prompt_folder_path="./templates/"):
        self.prompt_folder_path = prompt_folder_path
        self.prompts = {}
        self.role=None
        self.task=None
        self.format = None
        self.qna = None
        self.few_shot = None
        self.prompt = None

    def setup_generic(self, component_name, file_name, index=None, key=None):
        """
        프롬프트 Generic setup 
        """
        file_path = os.path.join(self.prompt_folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_prompts = json.load(f)
            self.prompts.update(loaded_prompts)

        if key is not None:
            if key in self.prompts:
                setattr(self, component_name, PromptTemplate(input_variables=["context"], template=self.prompts[key]))
            else:
                raise ValueError(f"{component_name.capitalize()} key '{key}' not found in the prompts.")

        elif index is not None:
            keys = list(self.prompts.keys())
            if 0 <= index < len(keys):
                selected_key = keys[index]
                setattr(self, component_name, PromptTemplate(input_variables=["context"], template=self.prompts[selected_key]))
            else:
                raise IndexError(f"{component_name.capitalize()} index {index} is out of range.")
        else:
            raise ValueError("Either 'key' or 'index' must be provided.")

    # setup templates
    def _setup_role(self, file_name="roles.json", index=None, key=None):
        """
        Set up role-specific prompt.
        """
        self.setup_generic("role", file_name, index, key)

    def _setup_task(self, file_name="tasks.json", index=None, key=None):
        """
        Set up task-specific prompt.
        """
        self.setup_generic("task", file_name, index, key)

    def _setup_format(self, file_name="formats.json", index=None, key=None):
        """
        Set up format-specific prompt.
        """
        self.setup_generic("format", file_name, index, key)
        
    def _setup_qna(self, file_name="qnas.json", index=None, key=None):
        """
        Set up QnA-specific prompt.
        """
        self.setup_generic("qna", file_name, index, key)

    def _setup_few_shot(self, file_name="few_shots.json", index=None, key=None):
        """
        Set up few-shot examples. 여기서 chain of thought, self consistency 설정 가능
        """
        self.setup_generic("few_shot", file_name, index, key)

    def generate_prompt(self, experiment_options='ex_options.json', option = "default"):
        """
        Generate a combined prompt incorporating role, task, format, and optional styles.
        """
        # experiment option에 따라서 _setup_함수들 로드
        experiment_file_path = os.path.join(self.prompt_folder_path, experiment_options)
        if not os.path.exists(experiment_file_path):
            raise FileNotFoundError(f"Experiment options file not found: {experiment_file_path}")

        with open(experiment_file_path, "r", encoding="utf-8") as f:
            options = json.load(f)
        
        options = options[option]

        # Load settings using _setup_ methods
        if "role" in options:
            self._setup_role(file_name="roles.json", key=options["role"])
        if "task" in options:
            self._setup_task(file_name="tasks.json", key=options["task"])
        if "format" in options:
            self._setup_format(file_name="formats.json", key=options["format"])
        if "qna" in options:
            self._setup_qna(file_name="qnas.json", key=options["qna"])
        if "few_shot" in options:
            self._setup_few_shot(file_name="fewshots.json", key=options["fewshot"])

        # Base components
        role_part = self.role.template if self.role else ""
        task_part = self.task.template if self.task else ""
        format_part = self.format.template if self.format else ""
        qna_part = self.qna.template if self.qna else ""
        few_shot_part = self.few_shot.template if self.few_shot else ""

        # Combine all parts
        combined_prompt = "\n".join(
            filter(None, [role_part, task_part, format_part, qna_part, few_shot_part])
        )
        self.prompt = combined_prompt
        return combined_prompt