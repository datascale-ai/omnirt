"""Resident worker abstractions."""

from omnirt.workers.managed import ManagedGrpcResidentWorkerProxy
from omnirt.workers.resident import ResidentModelWorker, ResidentWorkerHandle
from omnirt.workers.remote import GrpcResidentWorkerProxy, ResidentWorkerService

__all__ = [
    "GrpcResidentWorkerProxy",
    "ManagedGrpcResidentWorkerProxy",
    "ResidentModelWorker",
    "ResidentWorkerHandle",
    "ResidentWorkerService",
]
