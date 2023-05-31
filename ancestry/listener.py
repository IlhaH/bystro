import urllib
import argparse
from dataclasses import dataclass
import logging
import itertools
from pathlib import Path

from pystalk import BeanstalkClient, BeanstalkError
from pystalk.client import Job
from ruamel.yaml import YAML

from ancestry_types import AncestrySubmission, HostAndPort


logging.basicConfig(
    filename="ancestry.log",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Starting ancestry listener")

BEANSTALK_TIMEOUT_ERROR = "TIMED_OUT"


def _load_yaml(path: Path) -> dict[str, str | dict]:
    with open(path, "r", encoding="utf-8") as queue_config_file:
        return YAML(typ="safe").load(queue_config_file)


def execute_job(job: Job) -> None:
    msg = "executing job: {}".format(job)
    print(msg)
    logger.info(msg)
    json_payload = job.job_data.decode()
    ancestry_submission = AncestrySubmission.parse_raw(json_payload)
    print("parsed ancestry submission:", ancestry_submission)


def main():
    """
    Start ancestry server that listens to beanstalkd queue
    and renders global ancestry predictions
    """
    parser = argparse.ArgumentParser(description="Run the ancestry server.")
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
    )
    args = parser.parse_args()
    beanstalk_conf = _load_yaml(Path(args.queue_conf))["beanstalkd"]
    print(beanstalk_conf)
    addresses = beanstalk_conf["addresses"]
    ancestry_tubes = beanstalk_conf["tubes"]["ancestry"]
    submission_tube = ancestry_tubes["submission"]
    events_tube = ancestry_tubes["events"]

    # todo: refactor multiple client logic
    beanstalk_clients = []
    for address in addresses:
        parsed_address = HostAndPort.from_str(address)
        client = BeanstalkClient(parsed_address.host, parsed_address.port)
        client.watchlist = {submission_tube}
        beanstalk_clients.append(client)
    num_clients = len(beanstalk_clients)

    for client_idx in itertools.count():
        logger.debug("starting loop with {}".format(client_idx))
        client = beanstalk_clients[client_idx % num_clients]
        try:
            job = client.reserve_job()
        except BeanstalkError as err:
            if err.message == BEANSTALK_TIMEOUT_ERROR:
                logger.debug(
                    "Timed out while trying to reserve a job: this is expected if no jobs are present"
                )
                continue
            else:
                raise err
        try:
            result = execute_job(job)
        except Exception as e:
            logger.error("Encountered exception {} while handling job {}", e, job)
            client.release_job(job.job_id)
            raise
        client.delete_job(job.job_id)


if __name__ == "__main__":
    main()
