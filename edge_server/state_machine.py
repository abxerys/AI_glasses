"""State machine for edge-side user interaction modes."""

from enum import Enum


class Mode(str, Enum):
    NAVIGATION = "navigation"
    FIND_OBJECT = "find_object"
    TRANSLATION = "translation"


class GlassesStateMachine:
    def __init__(self) -> None:
        self.mode = Mode.NAVIGATION

    def switch_mode(self, next_mode: Mode) -> Mode:
        """Switch between navigation, find-object, and translation modes."""
        self.mode = next_mode
        return self.mode

    def current_mode(self) -> Mode:
        return self.mode
