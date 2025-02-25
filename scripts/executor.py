import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
from tqdm.auto import tqdm
from more_itertools import batched
from langchain.prompts.chat import ChatPromptTemplate

from scripts.lmm import LLM, ChatHistory
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
    model_output: ChatHistory
    execution_time: int

    def get_model_name(self) -> str:
        """Get the name of the model used"""
        return self.model_name

    def get_output(self) -> str:
        """Get the raw model output"""
        return self.model_output

    def get_execution_time(self) -> int:
        """Get execution duration in seconds"""
        return self.execution_time

    def get_response(self) -> ChatHistory:
        """Get the final model response"""
        return self.model_output


@dataclass
class Dataset:
    """Dataset of riddle questions"""

    name: str
    riddles: list[RiddleQuestion]


class Executor:
    """Executes language models on riddle questions"""

    def __init__(self, models: list[LLM]):
        self.models = {
            model.name: model for model in models
        }  # Convert to dict for O(1) lookup
        logger.info(f"Initialized executor with {len(models)} models.")
        self.results_dir = Path("results")

    async def _execute_riddle_base(
        self,
        model: LLM,
        riddle: RiddleQuestion,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
        is_async: bool = True,
    ) -> ExecutionResult:
        """Base execution function for both sync and async operations"""
        template_args = args_generator(riddle)
        start_time = time.time()
        output = (
            await model.agenerate(prompt_template, template_args)
            if is_async
            else model.generate(prompt_template, template_args)
        )
        delta = time.time() - start_time

        return ExecutionResult(
            model_name=model.name,
            riddle=riddle,
            model_output=output,
            execution_time=delta,
        )

    def _execute_riddle(self, *args, **kwargs):
        """Synchronous wrapper for _execute_riddle_base"""
        return self._execute_riddle_base(*args, **kwargs, is_async=False)

    async def _aexecute_riddle(self, *args, **kwargs):
        """Asynchronous wrapper for _execute_riddle_base"""
        return await self._execute_riddle_base(*args, **kwargs, is_async=True)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for file system use"""
        return "-".join(
            filter(
                None,
                "".join(c if c.isalnum() else "-" for c in name).lower().split("-"),
            )
        )

    def _get_paths(self, run_name: str | None) -> tuple[str, Path, Path]:
        """Get sanitized run name and relevant paths"""
        sanitized_run_name = self._sanitize_name(run_name or "default-run")
        run_dir = self.results_dir / sanitized_run_name
        return (
            sanitized_run_name,
            run_dir / "checkpoints",
            run_dir / f"{sanitized_run_name}_results.pkl",
        )

    async def _process_model(
        self,
        model: LLM,
        dataset: Dataset,
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        checkpoints_dir: Path,
        create_checkpoints: bool,
        resume_from_checkpoint: bool,
        batch_size: int = 4,
        suffix_name: str | None = None,
        is_async: bool = True,
    ) -> list[ExecutionResult]:
        """Process a single model with shared logic for sync/async execution"""
        checkpoint_path = (
            checkpoints_dir
            / f"{dataset.name}_{model.name}{'_' + suffix_name if suffix_name else ''}.pkl"
        )

        if resume_from_checkpoint and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                logger.info(f"Loading cached results from {checkpoint_path}")
                return pickle.load(f)

        model.load()
        riddles = dataset.riddles
        try:
            template = (
                prompt_template(model.name)
                if callable(prompt_template)
                else prompt_template
            )
            results = []

            if is_async:
                total_riddles = len(riddles)
                results = []
                with tqdm(total=total_riddles, desc=f"{model.name}") as pbar:
                    for chunk in batched(riddles, batch_size):
                        tasks = [
                            self._aexecute_riddle(
                                model, riddle, template, args_generator
                            )
                            for riddle in chunk
                        ]
                        chunk_results = await asyncio.gather(*tasks)
                        results.extend(chunk_results)
                        pbar.update(len(chunk))
            else:
                results = [
                    self._execute_riddle(model, riddle, template, args_generator)
                    for riddle in tqdm(riddles, desc=model.name)
                ]

            if create_checkpoints:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)

            return results
        finally:
            model.cleanup()

    async def execute(self, *args, **kwargs) -> dict[str, list[ExecutionResult]]:
        """Synchronous execution wrapper"""
        return await self._execute_base(*args, **kwargs, is_async=False)

    async def aexecute(self, *args, **kwargs) -> dict[str, list[ExecutionResult]]:
        """Asynchronous execution wrapper"""
        return await self._execute_base(*args, **kwargs, is_async=True)

    async def _execute_base(
        self,
        input_data: Dataset | list[Dataset],
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        run_name: str | None = None,
        suffix_name: str | None = None,
        dump_to_pickle: bool = False,
        create_checkpoints: bool = False,
        resume_from_checkpoint: bool = False,
        batch_size: int = 4,
        is_async: bool = True,
    ) -> dict[str, list[ExecutionResult]]:
        """Base execution logic for both sync and async operations"""
        run_name, checkpoints_dir, results_path = self._get_paths(run_name)
        sanitized_suffix_name = self._sanitize_name(suffix_name)

        if results_path.exists():
            with open(results_path, "rb") as f:
                logger.info(f"Loading cached results from {results_path}")
                return pickle.load(f).get("results", {})

        if not isinstance(input_data, list):
            input_data = [input_data]

        logger.info(f"Processing {len(input_data)} datasets")
        results = {}
        for dataset in input_data:
            logger.info(f"Processing dataset: {dataset.name}")
            results[dataset.name] = {
                f"{name}_{sanitized_suffix_name}": await self._process_model(
                    model,
                    dataset,
                    prompt_template,
                    args_generator,
                    checkpoints_dir,
                    create_checkpoints,
                    resume_from_checkpoint,
                    batch_size,
                    sanitized_suffix_name,
                    is_async,
                )
                for name, model in self.models.items()
            }

        if dump_to_pickle:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "wb") as f:
                pickle.dump(
                    {
                        "run_name": f"{run_name}",
                        "results": results,
                    },
                    f,
                )

        return results
