import os
import logging

from . import get_logger, get_root


def report_env(to_stdout=False):
    from git import Repo, InvalidGitRepositoryError
    from socket import gethostname
    from getpass import getuser

    logger = get_logger('env', basic_logging_params={'level': logging.INFO if to_stdout else logging.WARNING},
        manual_flush=False)

    logger.info(f'Running on {gethostname()} as {getuser()}')

    project_path = os.path.dirname(os.path.realpath(__file__))
    try:
        repo = Repo(project_path, search_parent_directories=True)
        active_branch = repo.active_branch
        latest_commit = repo.commit(active_branch)
        latest_commit_sha = latest_commit.hexsha
        latest_commit_sha_short = repo.git.rev_parse(latest_commit_sha, short=6)
        logger.info(f'We are on branch {active_branch} using commit {latest_commit_sha_short}')
    except InvalidGitRepositoryError:
        logger.info(f'{project_path} is not a git repo')

    logger.info(f'Saving data to {logger.logdir}')

    logger.logger.handlers.clear()