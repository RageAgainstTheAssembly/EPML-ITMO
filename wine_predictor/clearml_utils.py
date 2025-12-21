import inspect
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, TypeVar

from clearml import Task
from clearml.model import OutputModel

T = TypeVar("T")

logger = logging.getLogger(__name__)

DEFAULT_CLEARML_PROJECT = "EPML-ITMO/HW5-ClearML"
DEFAULT_CLEARML_TASK_TYPE = "training"


@dataclass(frozen=True)
class ClearMLTaskHandle:
    task: Task
    created_new_task: bool


def _safe_status_str(task: Task) -> str:
    try:
        s = task.get_status()
    except Exception:
        s = getattr(task, "status", "unknown")
    return str(s).lower()


def init_clearml_task(
    *,
    project_name: str,
    task_name: str,
    task_type: str = DEFAULT_CLEARML_TASK_TYPE,
    tags: Optional[Iterable[str]] = None,
    reuse_current_task: bool = True,
) -> ClearMLTaskHandle:
    """
    Create or reuse a ClearML Task.
    """
    current = Task.current_task()

    if reuse_current_task and current is not None:
        status = _safe_status_str(current)
        if all(x not in status for x in ("completed", "failed", "aborted", "stopped")):
            if tags:
                try:
                    current.add_tags(list(tags))
                except Exception:
                    logger.debug("ClearML: failed to add tags", exc_info=True)
            return ClearMLTaskHandle(task=current, created_new_task=False)

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        reuse_last_task_id=False,
    )
    if tags:
        try:
            if tags:
                task.add_tags(list(tags))
        except Exception:
            logger.debug("ClearML: failed to add tags", exc_info=True)
    return ClearMLTaskHandle(task=task, created_new_task=True)


def connect_hyperparams(task: Task, params: Dict[str, Any]) -> None:
    """
    Stores hyperparameters in ClearML. This is safe during execution.
    (Do not call after task is completed.)
    """
    if not params:
        return
    try:
        task.connect(params)
    except Exception:
        logger.debug("ClearML: failed to add params", exc_info=True)


def connect_configuration(task: Task, name: str, config: Any) -> None:
    """
    Store configuration object in ClearML.
    """
    try:
        task.connect_configuration(name=name, configuration=config)
    except Exception:
        logger.debug("ClearML: failed to add configuration", exc_info=True)


def upload_artifact(task: Task, name: str, artifact: Any) -> None:
    """
    Upload a local file / folder / dict / jsonable object as a task artifact.
    """
    try:
        task.upload_artifact(name=name, artifact_object=artifact, wait_on_upload=True)
    except Exception:
        logger.debug("ClearML: failed to add upload artifact", exc_info=True)


def report_scalars(
    task: Task, metrics: Dict[str, Any], series: str = "metrics", iteration: int = 0
) -> None:
    """
    Reports numeric metrics as scalars; non-numeric values get dumped as a JSON artifact.
    """
    logger = task.get_logger()

    non_numeric: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                logger.report_scalar(
                    title=k, series=series, value=float(v), iteration=iteration
                )
            except Exception:
                non_numeric[k] = v
        else:
            non_numeric[k] = v

    if non_numeric:
        upload_artifact(task, f"{series}_non_numeric", non_numeric)


def register_output_model(
    *,
    task: Task,
    model_path: Path,
    model_name: str,
    framework: str = "sklearn",
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Register a model in ClearML Model Registry.

    NOTE: OutputModel.update_weights uses `weights_filename`, NOT `weights_path`.
    """
    if not model_path.exists():
        return None

    try:
        out_model = OutputModel(
            task=task,
            name=model_name,
            framework=framework,
            tags=list(tags) if tags else None,
        )
        if metadata:
            for k, v in metadata.items():
                try:
                    out_model.set_metadata(
                        str(k), json.dumps(v) if not isinstance(v, str) else v
                    )
                except Exception:
                    logger.debug("ClearML: failed to add set metadata", exc_info=True)

        uri = out_model.update_weights(weights_filename=str(model_path))
        return uri
    except Exception:
        return None


def _filter_kwargs_for_callable(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Helper: pass only kwargs that exist in the callable signature.
    This makes our wrappers robust if train/eval signatures evolve.
    """
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        return kwargs


def clearml_stage(
    *,
    project_name: str = DEFAULT_CLEARML_PROJECT,
    task_name: str,
    task_type: str = DEFAULT_CLEARML_TASK_TYPE,
    tags: Optional[Sequence[str]] = None,
    reuse_current_task: bool = True,
    auto_close_if_created: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for train/evaluate/notify stages.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            tag_list = list(tags) if tags else None

            handle = init_clearml_task(
                project_name=project_name,
                task_name=task_name,
                task_type=task_type,
                tags=tag_list,
                reuse_current_task=reuse_current_task,
            )
            task = handle.task

            try:
                connect_hyperparams(task, dict(kwargs))
            except Exception:
                logger.debug("ClearML: failed to connect hyperparams", exc_info=True)

            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    task.flush_wait_for_uploads()
                except Exception:
                    logger.debug("ClearML: failed to flush uploads", exc_info=True)

                if handle.created_new_task and auto_close_if_created:
                    try:
                        task.close()
                    except Exception:
                        logger.debug("ClearML: failed to close task", exc_info=True)

        return wrapper

    return decorator


def clearml_run(
    *,
    project_name: Optional[str] = None,
    default_project_name: Optional[str] = None,
    project: Optional[str] = None,
    task_name: Optional[str] = None,
    default_task_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    **_ignored: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Backwards-compatible decorator used by train/evaluate/notify.
    """
    effective_project = (
        project_name
        or default_project_name
        or project
        or os.getenv("CLEARML_PROJECT_NAME")
        or "EPML-ITMO / HW5"
    )
    effective_task = (
        task_name or default_task_name or os.getenv("CLEARML_TASK_NAME") or "Run"
    )

    base_decorator = clearml_stage(
        project_name=effective_project,
        task_name=effective_task,
        tags=tags,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        wrapped = base_decorator(fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enable_clearml = kwargs.pop("enable_clearml", True)
            safe_kwargs = _filter_kwargs_for_callable(fn, kwargs)

            if not enable_clearml:
                return fn(*args, **safe_kwargs)

            return wrapped(*args, **safe_kwargs)

        return wrapper

    return decorator
