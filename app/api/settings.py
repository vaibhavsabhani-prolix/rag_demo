"""Application settings endpoints, stored in PostgreSQL."""

from fastapi import APIRouter

from app.api.schemas import UiSettings
from app.db import repository

router = APIRouter(prefix="/api/settings", tags=["settings"])

UI_KEY = "ui"


@router.get("/ui", response_model=UiSettings)
def get_ui_settings() -> UiSettings:
    # Missing or partial rows fall back to the defaults.
    return UiSettings(**(repository.get_setting(UI_KEY) or {}))


@router.put("/ui", response_model=UiSettings)
def save_ui_settings(settings: UiSettings) -> UiSettings:
    repository.set_setting(UI_KEY, settings.model_dump())
    return settings
