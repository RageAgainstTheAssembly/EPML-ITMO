import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
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


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x) and not isinstance(x, type):
        try:
            return asdict(x)
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def _tags_to_list(tags: Any) -> Optional[list[str]]:
    if tags is None:
        return None
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, dict):
        out = []
        for k, v in tags.items():
            out.append(f"{k}={v}")
        return out
    if isinstance(tags, (list, tuple, set)):
        return [str(x) for x in tags]
    return [str(tags)]


def init_clearml_task(
    *,
    project_name: str,
    task_name: str,
    task_type: str = DEFAULT_CLEARML_TASK_TYPE,
    tags: Optional[Iterable[str]] = None,
    reuse_current_task: bool = True,
) -> ClearMLTaskHandle:
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
            task.add_tags(list(tags))
        except Exception:
            logger.debug("ClearML: failed to add tags", exc_info=True)

    return ClearMLTaskHandle(task=task, created_new_task=True)


def connect_hyperparams(task: Task, params: Dict[str, Any]) -> None:
    if not params:
        return
    try:
        task.connect(_to_jsonable(params))
    except Exception:
        logger.debug("ClearML: failed to add params", exc_info=True)


def connect_configuration(task: Task, name: str, config: Any) -> None:
    try:
        task.connect_configuration(name=name, configuration=_to_jsonable(config))
    except Exception:
        logger.debug("ClearML: failed to add configuration", exc_info=True)


def upload_artifact(task: Task, name: str, artifact: Any) -> None:
    try:
        task.upload_artifact(name=name, artifact_object=artifact, wait_on_upload=True)
    except Exception:
        logger.debug("ClearML: failed to upload artifact", exc_info=True)


def report_scalars(
    task: Task,
    metrics: Dict[str, Any],
    series: str = "metrics",
    iteration: int = 0,
) -> None:
    lg = task.get_logger()

    non_numeric: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                lg.report_scalar(
                    title=str(k), series=series, value=float(v), iteration=iteration
                )
            except Exception:
                non_numeric[k] = v
        else:
            non_numeric[k] = v

    if non_numeric:
        upload_artifact(task, f"{series}_non_numeric", _to_jsonable(non_numeric))


def register_output_model(
    *,
    task: Task,
    model_path: Path,
    model_name: str,
    framework: str = "sklearn",
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
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
                        str(k),
                        json.dumps(_to_jsonable(v)) if not isinstance(v, str) else v,
                    )
                except Exception:
                    logger.debug("ClearML: failed to set metadata", exc_info=True)

        uri = out_model.update_weights(weights_filename=str(model_path))
        return uri
    except Exception:
        logger.debug("ClearML: model registration failed", exc_info=True)
        return None


def _filter_kwargs_for_callable(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        return kwargs


def _auto_log_stage_result(
    *,
    task: Task,
    result: Any,
    register_model_name: Optional[str],
    base_tags: Optional[list[str]],
) -> None:
    if not isinstance(result, dict):
        return

    merged_tags: list[str] = []
    if base_tags:
        merged_tags.extend(base_tags)

    returned_tags = _tags_to_list(result.get("tags"))
    if returned_tags:
        merged_tags.extend(returned_tags)

    if merged_tags:
        try:
            task.add_tags(sorted(set(merged_tags)))
        except Exception:
            logger.debug("ClearML: failed to add tags", exc_info=True)

    params = result.get("params")
    if isinstance(params, dict):
        connect_hyperparams(task, params)

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        report_scalars(task, metrics, series="metrics", iteration=0)
        upload_artifact(task, "metrics_dict", _to_jsonable(metrics))

    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        for name, obj in artifacts.items():
            if obj is None:
                continue
            try:
                p = Path(obj) if isinstance(obj, (str, Path)) else None
                if p is not None and p.exists():
                    upload_artifact(task, str(name), p)
                else:
                    upload_artifact(task, str(name), _to_jsonable(obj))
            except Exception:
                logger.debug(
                    "ClearML: failed to upload artifact '%s'", name, exc_info=True
                )

    if register_model_name:
        model_path = None
        if isinstance(artifacts, dict):
            mp = artifacts.get("model_path")
            if isinstance(mp, (str, Path)):
                model_path = Path(mp)

        if model_path is not None and model_path.exists():
            uri = register_output_model(
                task=task,
                model_path=model_path,
                model_name=register_model_name,
                framework="sklearn",
                tags=merged_tags,
                metadata={
                    "params": params if isinstance(params, dict) else {},
                    "metrics": metrics if isinstance(metrics, dict) else {},
                },
            )
            if uri:
                upload_artifact(task, "registered_model_uri", uri)


def clearml_stage(
    *,
    project_name: str = DEFAULT_CLEARML_PROJECT,
    task_name: str,
    task_type: str = DEFAULT_CLEARML_TASK_TYPE,
    tags: Optional[Sequence[str]] = None,
    reuse_current_task: bool = True,
    auto_close_if_created: bool = True,
    register_model_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
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
                connect_hyperparams(
                    task, {k: _to_jsonable(v) for k, v in kwargs.items()}
                )
            except Exception:
                logger.debug("ClearML: failed to connect hyperparams", exc_info=True)

            if "config" in kwargs:
                try:
                    connect_configuration(task, "config", kwargs["config"])
                except Exception:
                    logger.debug(
                        "ClearML: failed to connect configuration", exc_info=True
                    )

            try:
                result = fn(*args, **kwargs)
                try:
                    _auto_log_stage_result(
                        task=task,
                        result=result,
                        register_model_name=register_model_name,
                        base_tags=tag_list,
                    )
                except Exception:
                    logger.debug("ClearML: auto-log failed", exc_info=True)
                return result
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
    task_type: Optional[Any] = None,
    tags: Optional[Any] = None,
    default_tags: Optional[Any] = None,
    register_model_name: Optional[str] = None,
    reuse_current_task: bool = True,
    auto_close_if_created: bool = True,
    **_ignored: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
    effective_task_type = task_type or DEFAULT_CLEARML_TASK_TYPE

    merged_tags: list[str] = []
    for t in (_tags_to_list(default_tags), _tags_to_list(tags)):
        if t:
            merged_tags.extend(t)

    base_decorator = clearml_stage(
        project_name=effective_project,
        task_name=effective_task,
        task_type=effective_task_type,
        tags=merged_tags or None,
        reuse_current_task=reuse_current_task,
        auto_close_if_created=auto_close_if_created,
        register_model_name=register_model_name,
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
