"""Provide a listener to allow the ancestry model to talk over beanstalk."""
import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

from pystalk import BeanstalkClient, BeanstalkError
from pystalk.client import Job
from ruamel.yaml import YAML

from ancestry.ancestry_types import Address, AncestrySubmission

logging.basicConfig(
    filename="ancestry.log",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Starting ancestry listener")

BEANSTALK_TIMEOUT_ERROR = "TIMED_OUT"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file_handler:
        return YAML(typ="safe").load(file_handler)


def _execute_job(job: Job) -> None:
    """Represent dummy job that just extracts the vcf for now."""
    msg = f"executing job: {job}"
    logger.info("got message: %s", msg)
    json_payload = job.job_data.decode()
    ancestry_submission = AncestrySubmission.parse_raw(json_payload)
    logger.debug("parsed ancestry submission: %s", ancestry_submission)


def main() -> None:
    """Run ancestry server accepting genotype requests and rendering global ancestry predictions."""
    parser = argparse.ArgumentParser(description="Run the ancestry server.")
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
    )
    args = parser.parse_args()
    beanstalk_conf = _load_yaml(Path(args.queue_conf))["beanstalkd"]
    addresses: dict[str, Any] = beanstalk_conf["addresses"]
    ancestry_tubes: dict[str, Any] = beanstalk_conf["tubes"]["ancestry"]
    submission_tube = ancestry_tubes["submission"]
    _events_tube = ancestry_tubes["events"]

    # todo: refactor multiple client logic
    beanstalk_clients = []
    for address in addresses:
        parsed_address = Address.from_str(address)
        client = BeanstalkClient(parsed_address.host, parsed_address.port)
        client.watchlist = {submission_tube}
        beanstalk_clients.append(client)
    num_clients = len(beanstalk_clients)

    for client_idx in itertools.count():
        logger.debug("starting ancestry listening loop with %s", client_idx)
        client = beanstalk_clients[client_idx % num_clients]
        try:
            job = client.reserve_job()
        except BeanstalkError as err:
            if err.message == BEANSTALK_TIMEOUT_ERROR:
                logger.debug(
                    "Timed out while reserving a job: this is expected if no jobs are present"
                )
                continue
            raise
        try:
            _execute_job(job)
        except Exception:
            logger.exception("Encountered exception while handling job %s", job)
            client.release_job(job.job_id)
            raise
        client.delete_job(job.job_id)


if __name__ == "__main__":
    main()
