import numpy as np
import pytest

from src.routing.router import RoutingConfig


class TestRoutingConfig:
    def test_default_config(self):
        config = RoutingConfig()
        assert config.easy_threshold == 0.7
        assert config.hard_threshold == 0.6
        assert config.escalation_threshold == 0.5
        assert config.enable_cost_optimization is True

    def test_custom_config(self):
        config = RoutingConfig(easy_threshold=0.9, hard_threshold=0.8)
        assert config.easy_threshold == 0.9
        assert config.hard_threshold == 0.8
