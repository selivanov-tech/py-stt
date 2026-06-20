"""Python model-worker: a thin FastAPI service that turns audio into text.

It holds all STT engines (local + cloud) and is called by the Go service. It is
deliberately dumb and stateless — the queue, jobs, and API live in Go.
"""
