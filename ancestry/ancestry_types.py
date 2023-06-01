"""Pydantic validators for common shapes of data in ancestry."""

import re
import urllib
from ipaddress import AddressValueError, IPv4Address
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Extra,
    Field,
    confloat,
    validator,
)

MAX_VALID_PORT = 2**16 - 1


class Address(BaseModel, extra=Extra.forbid):
    """Model a host and port."""

    host: str
    port: int = Field(ge=0, le=MAX_VALID_PORT)

    @classmethod
    def from_str(cls: type["Address"], raw_address: str) -> "Address":
        """Parse an Address from a string."""
        # urlsplit needs an absolute url, so we need to make it absolute if not so already.
        if not re.match(r"(beanstalkd:)?//", raw_address):
            absolute_address = "//" + raw_address
        else:
            absolute_address = raw_address
        # mypy seems confused about urllib's attrs
        parsed_address = urllib.parse.urlsplit(absolute_address)  # type: ignore [attr-defined]
        return cls(
            host=parsed_address.hostname,
            port=parsed_address.port,
        )

    @validator("host")
    def host_is_valid_ip_address(cls: "Address", raw_host: str) -> str:  # noqa: [N805]
        """Ensure host is a valid IPv4 address."""
        #  We really shouldn't be storing IP addresses as raw strings at all, but this usage is
        #  ubiquitous in libraries we need to work with.  A compromise is to validate the address
        #  string upon receipt and store it as str afterwards.
        try:
            host = IPv4Address(raw_host)
        except AddressValueError as err:
            err_msg = f"Couldn't parse host ({raw_host}) as IPv4Address: {err}"
            raise ValueError(err_msg) from err
        return str(host)


class AncestrySubmission(BaseModel, extra=Extra.forbid):
    """Represent an incoming submission to the ancestry worker."""

    vcf_path: str


class ProbabilityInterval(BaseModel, extra=Extra.forbid):
    """Represent an interval of probabilities."""

    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)

    @validator("upper_bound")
    def interval_is_valid(
        cls: "ProbabilityInterval",  # noqa: N805
        upper_bound: float,
        values: dict[str, Any],
    ) -> float:
        """Ensure interval is non-empty."""
        lower_bound = values["lower_bound"]
        if not lower_bound <= upper_bound:
            err_msg = (
                f"Must have lower_bound <= upper_bound:"
                f" got (lower_bound={lower_bound}, upper_bound={upper_bound}) instead."
            )
            raise ValueError(err_msg)
        return upper_bound


# NB: We might consider that a vector of ProbabilityIntervals should
# have additional validation properties, like that the sums of the
# lower bounds, upper bounds, or midpoints should be close to one.
# But constraints on the bounds don't hold in general (consider the
# vector of intervals [(0.4, 0.6), (0.4, 0.6)]), and we can't know how
# well the midpoints of the intervals reflect the point estimate in
# general, so we'll punt on this and assume it's the ML model's
# responsibility to give us scientifically sensible results.


# this definition is mildly ugly but the alternative is to
# generate it dynamically, which would be even worse...
class PopulationVector(BaseModel, extra=Extra.forbid):
    """A vector of probability intervals for populations."""

    ACB: ProbabilityInterval
    ASW: ProbabilityInterval
    BEB: ProbabilityInterval
    CDX: ProbabilityInterval
    CEU: ProbabilityInterval
    CHB: ProbabilityInterval
    CHS: ProbabilityInterval
    CLM: ProbabilityInterval
    ESN: ProbabilityInterval
    FIN: ProbabilityInterval
    GBR: ProbabilityInterval
    GIH: ProbabilityInterval
    GWD: ProbabilityInterval
    IBS: ProbabilityInterval
    ITU: ProbabilityInterval
    JPT: ProbabilityInterval
    KHV: ProbabilityInterval
    LWK: ProbabilityInterval
    MAG: ProbabilityInterval
    MSL: ProbabilityInterval
    MXL: ProbabilityInterval
    PEL: ProbabilityInterval
    PJL: ProbabilityInterval
    PUR: ProbabilityInterval
    STU: ProbabilityInterval
    TSI: ProbabilityInterval
    YRI: ProbabilityInterval


class SuperpopVector(BaseModel, extra=Extra.forbid):
    """A vector of probability intervals for superpopulations."""

    AFR: ProbabilityInterval
    AMR: ProbabilityInterval
    EAS: ProbabilityInterval
    EUR: ProbabilityInterval
    SAS: ProbabilityInterval


class AncestryResult(BaseModel, extra=Extra.forbid):
    """An ancestry result from a sample."""

    sample_id: str
    populations: PopulationVector
    superpops: SuperpopVector
    missingness: Annotated[
        float,
        confloat(
            ge=0,
            le=1,
        ),
    ]


class AncestryResponse(BaseModel, extra=Extra.forbid):
    """An outgoing response from the ancestry worker."""

    vcf_filepath: str
    results: list[AncestryResult]
