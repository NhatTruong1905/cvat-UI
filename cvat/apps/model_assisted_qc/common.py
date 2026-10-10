from rest_framework.exceptions import NotFound, PermissionDenied

from cvat.apps.engine.models import Task
from cvat.apps.engine.permissions import TaskPermission


def get_task(request, task_id):
    try:
        task = Task.objects.select_related("organization").get(pk=task_id)
    except Task.DoesNotExist as exc:
        raise NotFound("Task was not found") from exc
    if not TaskPermission.create_scope_view(request, task).check_access().allow:
        raise PermissionDenied("You do not have access to this task")
    return task
