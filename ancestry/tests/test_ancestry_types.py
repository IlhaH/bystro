"""Test ancestry_types.py."""

from ipaddress import AddressValueError

import pytest
from pydantic import ValidationError

from ancestry.ancestry_types import Address, PopulationVector, ProbabilityInterval, SuperpopVector

POPULATIONS = [
    "ACB",
    "ASW",
    "BEB",
    "CDX",
    "CEU",
    "CHB",
    "CHS",
    "CLM",
    "ESN",
    "FIN",
    "GBR",
    "GIH",
    "GWD",
    "IBS",
    "ITU",
    "JPT",
    "KHV",
    "LWK",
    "MAG",
    "MSL",
    "MXL",
    "PEL",
    "PJL",
    "PUR",
    "STU",
    "TSI",
    "YRI",
]

SUPERPOPS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS",
]


def test_Address() -> None:
    """Ensure we can instantiate, validate Address correctly."""
    well_formed_addresses = [
        "127.0.0.1:80",
        "//127.0.0.1:80",
        "beanstalkd://127.0.0.1:80",
    ]
    for raw_address in well_formed_addresses:
        address = Address.from_str(raw_address)
        assert address.host == "127.0.0.1"
        assert address.port == 80

    malformed_addresses = [
        "127.0.0.0.1:80",
        "127.0.1:80",
        "127.0.0.1",
        "///127.0.0.1:80",
        "beanstalkd//127.0.0.1:80",
    ]
    for raw_address in malformed_addresses:
        with pytest.raises((ValidationError, AddressValueError)):
            address = Address.from_str(raw_address)


def test_ProbabilityInterval() -> None:
    """Ensure we can instantiate, validate ProbabilityInterval correctly."""
    ProbabilityInterval(lower_bound=0.1, upper_bound=0.9)

    with pytest.raises(ValidationError):
        ProbabilityInterval(lower_bound=0.1, upper_bound=1.1)

    with pytest.raises(ValidationError):
        ProbabilityInterval(lower_bound=-2, upper_bound=1.1)

    with pytest.raises(ValidationError):
        ProbabilityInterval(lower_bound=1, upper_bound=0)


prob_int = ProbabilityInterval(lower_bound=0, upper_bound=1)
pop_kwargs = {pop: prob_int for pop in POPULATIONS}


def test_PopulationVector() -> None:
    """Ensure we can instantiate, validate PopulationVector correctly."""
    PopulationVector(**pop_kwargs)


def test_PopulationVector_with_missing_key() -> None:
    pop_kwargs_with_missing_key = pop_kwargs.copy()
    del pop_kwargs_with_missing_key["ACB"]
    with pytest.raises(ValidationError):
        PopulationVector(**pop_kwargs_with_missing_key)


def test_PopulationVector_with_extra_key() -> None:
    pop_kwargs_with_extra_key = pop_kwargs.copy()
    pop_kwargs_with_extra_key["FOO"] = prob_int
    with pytest.raises(ValidationError):
        PopulationVector(**pop_kwargs_with_extra_key)


def test_SuperpopVector() -> None:
    prob_int = ProbabilityInterval(lower_bound=0, upper_bound=1)
    SuperpopVector(
        AFR=prob_int,
        AMR=prob_int,
        EAS=prob_int,
        EUR=prob_int,
        SAS=prob_int,
    )


def test_SuperpopVector_missing_key() -> None:
    with pytest.raises(ValidationError):
        SuperpopVector(
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
        )


def test_SuperpopVector_extra_key() -> None:
    with pytest.raises(ValidationError):
        SuperpopVector(
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
            SAS=prob_int,
            FOO=prob_int,
        )
