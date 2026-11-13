"""Live (real Groq) execution path for The Relay.

Everything in this package talks to the real Groq API. Nothing here simulates a
model: latency, limits, usage and errors all come from real responses. Modules
are written so the pure logic (parsing, budgeting, gating, caching, evals) can
be unit-tested without network access or pydantic.
"""
