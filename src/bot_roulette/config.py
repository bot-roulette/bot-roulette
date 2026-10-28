"""Library settings: what persona creation needs, read from a YAML file.

The models are strict: an unknown key (a typo), a missing key or a wrong
value is an error naming it, instead of surfacing halfway through a paid
creation job. There are no hidden defaults here: the file is the only
source of these values. Nothing is loaded at import; call load_config.

An application with settings of its own keeps them in the same file by
subclassing Config and passing it as ``schema`` (the web app does).
"""

import re
from pathlib import Path
from typing import Annotated, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

NonEmpty = Annotated[list[str], Field(min_length=1)]


class _Section(BaseModel):
    # Unknown keys are errors: a misspelled setting must not be silently
    # ignored. Frozen: code can't change a setting by accident.
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreationConfig(_Section):
    interview_questions: PositiveInt
    voice_partners: PositiveInt
    voice_turns: PositiveInt
    voice_min_turns: PositiveInt

    @model_validator(mode="after")
    def _min_turns_reachable(self):
        if self.voice_min_turns > self.voice_turns:
            raise ValueError("voice_min_turns can't be more than voice_turns")
        return self


_AGE_BAND_RE = re.compile(r"^(\d{1,3})-(\d{1,3})$")


class RandomPersonasConfig(_Section):
    # The background of a random persona. Who they are (name, sex,
    # country) is chosen when creating them: see creation.PersonaIdentity.
    age_bands: NonEmpty
    education: NonEmpty
    work_areas: NonEmpty
    settlements: NonEmpty
    living_situations: NonEmpty
    mobility: NonEmpty
    employment_stability: NonEmpty
    life_events: NonEmpty

    @field_validator("age_bands")
    @classmethod
    def _age_band_format(cls, bands: list[str]) -> list[str]:
        for band in bands:
            m = _AGE_BAND_RE.match(band)
            if not m or int(m[1]) > int(m[2]):
                raise ValueError(f'age band {band!r} must look like "25-34"')
        return bands


class Config(_Section):
    creation: CreationConfig
    random_personas: RandomPersonasConfig


ConfigT = TypeVar("ConfigT", bound=Config)


def load_config(path: str | Path, schema: type[ConfigT] = Config) -> ConfigT:
    """Read and check a config file against ``schema`` (Config, or an
    application's subclass of it); RuntimeError lists every problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"{path.name} not found at {path}") from None
    except yaml.YAMLError as e:
        raise RuntimeError(f"{path.name} is not valid YAML: {e}") from None
    try:
        return schema.model_validate(raw or {})
    except ValidationError as e:
        problems = "\n".join(
            f"  {'.'.join(str(p) for p in err['loc']) or '(top level)'}: "
            f"{err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        )
        raise RuntimeError(f"{path.name} is invalid:\n{problems}") from None
