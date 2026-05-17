"""Gemini API client for semantic understanding and translation."""

from typing import Dict

import requests


class GeminiAgent:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def ask(self, prompt: str) -> Dict:
        """Send text prompt to Gemini model and return raw response."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(url, headers=headers, params=params, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
