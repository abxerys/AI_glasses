"""Google Maps API client for route navigation."""

from typing import Dict

import requests


class NavigationAPI:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_route(self, origin: str, destination: str, mode: str = "walking") -> Dict:
        """Request route data from Google Directions API."""
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": self.api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
