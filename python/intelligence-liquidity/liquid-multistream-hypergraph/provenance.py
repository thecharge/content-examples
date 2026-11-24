"""Device incarnations and bounded derived features. Targets remain external."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Sensor:
    field: str
    source: str
    incarnation: str
    room: str
    schema: str
    calibration: str
    normalization: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in asdict(self).values()):
            raise ValueError("Every sensor provenance field is required")


def observation(event_id, event_time, arrival_time, values, sensors, quality, context):
    if (
        not all(math.isfinite(t) for t in (event_time, arrival_time))
        or arrival_time < event_time
    ):
        raise ValueError("Arrival time must follow finite event time")
    if not isinstance(context, str) or not context:
        raise ValueError("Observation context required")
    if set(values) != set(sensors) or set(quality) - set(values):
        raise ValueError("Every reading needs sensor provenance")
    readings, metadata = {}, {}
    for field, value in values.items():
        sensor = sensors[field]
        if not isinstance(sensor, Sensor) or sensor.field != field:
            raise ValueError("Sensor field mismatch")
        status = quality.get(field, "missing" if value is None else "good")
        if status not in ("good", "missing", "broken", "stale"):
            raise ValueError("Unknown sensor quality")
        value = None if value is None else float(value)
        if value is not None and not math.isfinite(value):
            raise ValueError("Use null and a quality marker for invalid readings")
        readings[field] = value if status == "good" else None
        metadata[field] = dict(asdict(sensor), quality=status, raw=value)
    return dict(
        id=event_id,
        event_time=event_time,
        arrival_time=arrival_time,
        context=context,
        values=readings,
        sensors=metadata,
    )


class Features:
    """Explicit transforms with ancestry. Their weights still need labeled fitting."""

    def __init__(self, maximum=8):
        if maximum < 1:
            raise ValueError("Feature budget must be positive")
        self.maximum, self.definitions = maximum, {}

    def add(self, name, sources, evidence_id, operation="mean"):
        if name in self.definitions or len(self.definitions) >= self.maximum:
            raise ValueError("Feature identity or budget exhausted")
        if (
            operation not in ("mean", "difference")
            or not sources
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("Invalid derived feature")
        if operation == "difference" and len(sources) != 2:
            raise ValueError("Difference needs two sources")
        if name in sources or any(s in self.definitions for s in sources):
            raise ValueError("Derived feature chains are disabled to bound work")
        if not evidence_id:
            raise ValueError("Feature needs decision provenance")
        self.definitions[name] = dict(
            sources=list(sources), evidence=evidence_id, operation=operation
        )

    def apply(self, values):
        result = dict(values)
        for name, item in self.definitions.items():
            if name in values:
                raise ValueError("Derived feature would overwrite a sensor")
            source = [values.get(s) for s in item["sources"]]
            if any(v is None for v in source):
                result[name] = None
            elif item["operation"] == "mean":
                result[name] = sum(source) / len(source)
            else:
                result[name] = source[0] - source[1]
        return result
