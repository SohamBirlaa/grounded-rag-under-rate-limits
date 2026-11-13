"""Pytest configuration and environment fixtures for The Relay."""

import os

# Enforce offline mode for HuggingFace / FastEmbed to prevent network hangs in sandboxed/offline environments
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

