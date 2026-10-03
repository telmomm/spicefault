"""Workers for the engine tests. They live in a module so that worker processes can import them."""

from pathlib import Path

from spicefault.experiments import sample_stream


def draw(task, context):
    """Row and waveform that depend only on the task; one task in ten has no waveform."""
    index, replica = task
    rng = sample_stream(context["seed"], index, replica)
    if context.get("log"):
        (Path(context["log"]) / f"{index}_{replica}").touch()
    row = {"index": index, "replica": replica, "x": rng.normal(), "sim_ok": replica % 10 != 9}
    return row, rng.normal(size=context["n_points"]) if row["sim_ok"] else None


def scalar_only(task, context):
    return {"x": float(task) * context}, None


def fails(task, context):
    return {"x": 1 / task}, None
