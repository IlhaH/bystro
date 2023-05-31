import re
import numpy as np
from typing import Tuple
from ipaddress import IPv4Address
from dataclasses import dataclass
from pydantic import (
    BaseModel,
    Extra,
    confloat,
    validator,
    ValidationError,
    root_validator,
)
import urllib


class Address(BaseModel, extra=Extra.forbid):
    host: str
    port: int

    @classmethod
    def from_str(cls, raw_address: str) -> "Address":
        """Parse an Address from a string."""
        # urlsplit needs an absolute url...
        if not re.match(r"(beanstalkd:)?//", raw_address):
            absolute_address = "//" + raw_address
        else:
            absolute_address = raw_address
        parsed_address = urllib.parse.urlsplit(absolute_address)
        validated_host = str(IPv4Address(parsed_address.hostname))
        return cls(host=validated_host, port=parsed_address.port)

    @validator("host")
    def host_is_valid_ip_address(cls, raw_host):
        try:
            host = IPv4Address(raw_host)
        except AddressValueError as err:
            err_msg = f"Couldn't parse host ({raw_host}) as IPv4Address: {err}"
            raise ValueError(err_msg)
        return str(host)


class AncestrySubmission(BaseModel, extra=Extra.forbid):
    vcf_path: str


# ConstrainedTuple = Tuple[confloat(ge=0, le=1), confloat(ge=0, le=1)]


class ProbabilityInterval(BaseModel, extra=Extra.forbid):
    lower_bound: confloat(ge=0, le=1)
    upper_bound: confloat(ge=0, le=1)

    @validator("upper_bound")
    def interval_is_valid(cls, upper_bound, values):
        # upper_bound = v
        lower_bound = values["lower_bound"]
        if not lower_bound <= upper_bound:
            err_msg = "Must have lower_bound <= upper_bound: got (lower_bound={}, upper_bound={}) instead.".format(
                lower_bound, upper_bound
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
    AFR: ProbabilityInterval
    AMR: ProbabilityInterval
    EAS: ProbabilityInterval
    EUR: ProbabilityInterval
    SAS: ProbabilityInterval


class AncestryResult(BaseModel, extra=Extra.forbid):
    sample_id: str
    populations: PopulationVector
    superpops: SuperpopVector
    missingness: confloat(ge=0, le=1)


class AncestryResponse(BaseModel, extra=Extra.forbid):
    vcf_filepath: str
    results: list[AncestryResult]
