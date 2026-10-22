"""Draft, improvement, and repair prompts used by the search agent.

Prompt text and field ordering are part of the experiment contract. Keep edits
here separate from readability changes so cached/replayed requests remain valid.
"""

import random
import time
from typing import Any

import humanize

from .journal import Node
from .utils.response import trim_long_string, wrap_code


def format_time(time_in_sec: int | float) -> str:
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


def _prompt_opening(obfuscate: bool) -> str:
    return (
        "You are an expert machine learning engineer attempting a task. "
        if obfuscate
        else "You are a Kaggle grandmaster attending a competition. "
    )


class GenerationPromptsMixin:
    """Render prompts from state initialized by Agent.

    Reads cfg/acfg, task_desc, journal, data_preview, and the step/time budget.
    It owns no state and makes no model calls.
    """

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!)"
        }
        # tensorflow denying prompt since model couldn't debug tensorflow
        env_prompt["Installed Packages"] += " TensorFlow is NOT installed; use PyTorch instead."
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_remaining = self.deadline - time.time()
        steps_remaining = self.acfg.steps - self.current_step
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        metric_phrase = (
            f"the **{self.acfg.eval_metric}** metric (the competition's official "
            f"evaluation metric, as defined in the Task description)"
            if self.acfg.eval_metric
            else "the evaluation metric"
        )
        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {steps_remaining}>",
            f"The code should **implement the proposed solution** and **print the value of {metric_phrase} computed on a hold-out validation set**",
            "DO NOT display progress bars. If you use a function or library with integrated progress bars, disable them or pass the appropriate parameter to silence them.",
            "The validation score MUST be a genuine hold-out estimate: hold out a split the model never trains on, and do NOT refit the final model on the full data (including the validation rows) before scoring it — that resubstitution inflates the score and will not match the graded test performance.",
            "• Your local validation must evaluate the SAME core task as the final submission, not an easier proxy sub-problem.",
            "• ❌ FORBIDDEN: Silently redefining the task into a simpler objective (for example: multi-target -> single-target, detection -> presence-only classification, ranking -> plain classification, structured prediction -> one-field prediction).",
            "• The validation target, prediction format, and post-processing logic must stay semantically aligned with the required submission format.",
            "Use one shared prediction function for held-out validation and test submission, including numerical preprocessing, model eval mode, TTA and output aggregation. Score that complete prediction path on held-out rows before any optional full-data refit. Apply image normalization once per view; invert normalization before converting a standardized tensor back to an image. A submission-only inference change must be checked through the same validation prediction function.",
            "• In code and logs, make the validation setup easy to audit: split method, metric formula, predicted target, and any threshold/post-processing used.",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            "All data is under `./input`, but layouts vary by task. Inspect Data Overview or list `./input` first, then use the exact paths found there. Do not assume `input/train` or `input/test` exist; data may be flat files like `train.csv`/`test.json` or nested task-specific dirs. Never use `./data`.",
            "**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the ./submission/ directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!",
            "If a `sample_submission.csv` file exists, load it and use it as the template for your `submission.csv` to ensure the output format is correct.",
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
        ]

        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def _draft_prompt(self) -> dict:
        goal = "complete this task" if self.acfg.obfuscate else "win this competition"
        introduction = _prompt_opening(self.acfg.obfuscate) + (
            f"In order to {goal}, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                (
                    f"Validate using the competition's official evaluation metric — "
                    f"**{self.acfg.eval_metric}** — exactly as defined in the Task "
                    f"description; do not substitute a different or proxy metric."
                    if self.acfg.eval_metric
                    else "Propose an evaluation metric that is reasonable for this task."
                ),
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        return prompt

    def _improve_prompt(self, parent_node: Node) -> dict:
        introduction = _prompt_opening(self.acfg.obfuscate) + (
            "You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": parent_node.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Design": parent_node.plan or "",
            "Code": wrap_code(parent_node.code),
            "Results": trim_long_string(parent_node.analysis or ""),
            "Evidence source": (
                "Real execution" if parent_node.real_won else "World-model prediction"
            ),
        }
        if parent_node.metric is not None and parent_node.metric.value is not None:
            prompt["Previous solution"]["Validation Metric"] = str(parent_node.metric.value)
            if parent_node.metric.maximize is not None:
                prompt["Previous solution"]["Metric direction"] = (
                    "higher is better" if parent_node.metric.maximize else "lower is better"
                )
        prompt["Execution output"] = wrap_code(parent_node.term_out, lang="")

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement. Your improvement must be distinctly different from existing attempts in the Memory section",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
                "Don't suggest to do hyperparameter optimization, you should use the best hyperparameters from the previous solution.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        return prompt

    def _build_debug_prompt(
        self,
        prev_code: str,
        prev_term_out: str,
        prev_analysis: str = "",
        failure_reasons: dict[str, list[str]] | None = None,
    ) -> dict:
        """Route repair guidance by recorded failures, retaining legacy diagnoses."""
        introduction = _prompt_opening(self.acfg.obfuscate) + (
            "Your previous solution was flagged for repair. Use the recorded diagnosis "
            "and execution evidence below to identify and fix the reported defects. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(prev_code),
            "Execution output": wrap_code(prev_term_out, lang=""),
        }
        if prev_analysis and prev_analysis.strip():
            prompt["Diagnosis of the failure (from an expert reviewer)"] = trim_long_string(
                prev_analysis
            )
        categories = {category for category, reasons in (failure_reasons or {}).items() if reasons}
        if categories:
            prompt["Recorded failure categories"] = ", ".join(sorted(categories))
        prompt["Repair policy"] = (
            "Repair the previous implementation with the smallest changes required by the "
            "recorded defects, preserving unrelated working code. Successful execution "
            "does not establish that prediction, export, or evaluation logic is correct. "
            "A program that completed and wrote a CSV can still have a submission defect; "
            "those facts alone do not establish leakage or a metric error."
        )
        repair_guidance = []
        if "execution" in categories:
            repair_guidance.append(
                "Repair the recorded execution failure using the exception, resource limit, "
                "and relevant source operations. Distinguish observed failures from "
                "world-model predictions. Re-run to obtain any measurements or artifacts "
                "that the failure prevented."
            )
        if "submission_pipeline" in categories:
            repair_guidance.append(
                "Repair the diagnosed validation/submission prediction mismatch. Trace the "
                "specific preprocessing, inference mode, input features, row IDs, or class "
                "mapping named in the diagnosis and make the targeted correction. Recompute "
                "validation through the repaired prediction/export path and regenerate the CSV."
            )
        if "submission" in categories:
            repair_guidance.append(
                "Repair the diagnosed CSV creation or format defect against the task's "
                "submission requirements: the expected file, columns, row IDs, coverage, "
                "and prediction values. Preserve required predictions and verify the corrected CSV."
            )
        if categories & {"submission", "submission_pipeline"} and "metric" not in categories:
            repair_guidance.append(
                "Preserve the validation split and official metric definition for this "
                "submission repair; no separate metric defect was recorded. Change them "
                "only if you establish an additional concrete defect from the source."
            )
        if "metric" in categories:
            repair_guidance.append(
                "Repair the specific reported validation/metric defect. For demonstrated "
                "contamination, remove the prohibited target flow and recompute a held-out "
                "score. For a formula or scale defect, correct the official metric calculation. "
                "For a missing or unparseable metric, compute and report the required value. "
                "Do not merely clamp, hide, or re-print an invalid score. Ordinary early stopping "
                "and a final full-data refit after storing the held-out score are permitted "
                "and should be preserved unless a separate defect is demonstrated."
            )
        if not categories or categories - {
            "execution",
            "submission",
            "submission_pipeline",
            "metric",
        }:
            repair_guidance.append(
                "Establish the cause from the recorded diagnosis, source, and execution "
                "evidence before editing. Missing failure categories or an unavailable "
                "review do not establish a validation, prediction, or execution defect."
            )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
                "When the bug is a data-related error, consult the Data Overview section first for the actual data structure and format rather than assuming the cause.",
                *repair_guidance,
                "Before writing the final program, state the root cause and every intended change relative to the previous implementation, then implement only that plan. If you conclude the previous implementation cannot be repaired and must be replaced, say so explicitly and give the reason.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        return prompt
