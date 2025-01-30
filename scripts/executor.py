import logging
from dataclasses import dataclass
from collections.abc import Callable

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from scripts.lmm import LLM
from scripts.dataset import RiddleQuestion

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a model on a single riddle question"""

    model_name: str
    riddle: RiddleQuestion
    prompt_used: PromptValue
    model_output: str
    execution_time: int

    def get_model_name(self) -> str:
        """Get the name of the model used"""
        return self.model_name

    def get_prompt(self) -> PromptValue:
        """Get the prompt that was used"""
        return self.prompt_used

    def get_output(self) -> str:
        """Get the raw model output"""
        return self.model_output

    def get_execution_time(self) -> int:
        """Get execution duration in seconds"""
        return self.execution_time

    def get_response(self) -> str:
        """Get the final model response"""
        return self.model_output


class Executor:
    """Executes language models on riddle questions"""

    def __init__(self, models: list[LLM], riddle_dataset: list[RiddleQuestion]):
        """Initialize executor with models and dataset"""
        self.models = models
        self.riddle_dataset = riddle_dataset
        logger.info(
            f"Initialized executor with {len(models)} models and {len(riddle_dataset)} riddles"
        )

    def _execute_riddle(
        self,
        model: LLM,
        riddle: RiddleQuestion,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> ExecutionResult:
        """Execute model on a single riddle"""
        template_args = args_generator(riddle)
        output = model.generate(prompt_template, template_args)
        return ExecutionResult(
            model_name=model.name,
            riddle=riddle,
            prompt_used=template_args,
            model_output=output,
        )

    def execute(
        self,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> dict[str, list[ExecutionResult]]:
        """Execute all models on the dataset"""
        logger.info("Starting execution")
        results: dict[str, list[ExecutionResult]] = {
            model.name: [] for model in self.models
        }

        for model in self.models:
            logger.info(f"Processing {model.name}")
            try:
                model_results = []
                for i, riddle in enumerate(self.riddle_dataset, 1):
                    logger.debug(
                        f"Processing riddle {i}/{len(self.riddle_dataset)}: {riddle.id}"
                    )
                    result = self._execute_riddle(
                        model,
                        riddle,
                        prompt_template,
                        args_generator,
                    )
                    model_results.append(result)
                results[model.name] = model_results
                logger.info(f"Completed {model.name}")
            finally:
                model.cleanup()

        logger.info("Execution completed")
        return results
