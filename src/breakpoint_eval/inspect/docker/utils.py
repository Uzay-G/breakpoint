import pathlib
import shutil
import tempfile

import breakpoint_eval.problem

from .constants import BREAKPOINT_INSPECT_DOCKER_DIR


def get_compose_file_path(problem: breakpoint_eval.problem.Problem) -> pathlib.Path:
    """
    Get the Docker Compose file path for the given problem.
    """

    compose_path = (
        BREAKPOINT_INSPECT_DOCKER_DIR
        / ".dynamic-compose-files-for-inspect"
        / problem.id
    )
    compose_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(f"""\
services:
  # For the agent
  default:
    image: {problem.docker_image_name}
    command: sleep infinity
    network_mode: "none"  # This completely disables networking
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 4G
          pids: 8192
        reservations:
          cpus: '0.25'
          memory: 512M

  # For the submission grader
  submission_grader:
    image: {problem.docker_image_name}
    command: sleep infinity
    network_mode: "none"  # This completely disables networking
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 4G
          pids: 8192
        reservations:
          cpus: '0.25'
          memory: 512M
""")
        temp_name = tmp.name

    # After context exit, file still exists due to delete=False
    shutil.move(temp_name, compose_path)

    return compose_path
