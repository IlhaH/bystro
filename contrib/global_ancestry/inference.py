"""Perform inference for ancestry model.

Given an s3 filepath pointing to a VCF file, return a json object
containing purported fractions of population- and superpopulation
level ancestry for each sample.

"""

import logging
import sys
from pathlib import Path
from typing import TypeAlias

import allel
import numpy as np
import panda as pd
import skops.io as sio
from allel.stats.decomposition import GenotypePCA
from asserts import assert_equal, assert_equals, assert_true
from sklearn.ensemble import RandomForestClassifier
from utils import get_variant_ids_from_callset

logger = logging.getLogger(__name__)

PCAorRFC: TypeAlias = GenotypePCA | RandomForestClassifier

ANCESTRY_MODEL_PRODUCTS_DIR = Path("ancestry_model_products")
POP_RFC_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "pop_rfc.skops"
SUPERPOP_RFC_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "superpop_rfc.skops"
PCA_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "pca.skops"
VARIANTS_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "variants.txt"
POPULATIONS_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "populations.txt"
SUPERPOPS_FILEPATH = ANCESTRY_MODEL_PRODUCTS_DIR / "superpops.txt"

MISSING_FRACTION_THRESHOLD = 0.01


def _load_skops(filepath: Path, expected_types: list[str]) -> PCAorRFC:
    untrusted_types = sio.get_untrusted_types(file=filepath)
    assert_equals(
        "untrusted types in skops file",
        untrusted_types,
        "expected types",
        expected_types,
        comment=(
            "If these types don't match, the skops file is compromised and should not be loaded.  "
            "See: https://skops.readthedocs.io/en/stable/persistence.html for additional info."
        ),
    )
    return sio.load(file=filepath, trusted=expected_types)


def _load_pop_rfc() -> RandomForestClassifier:
    """Load RFC from disk."""
    expected_types = ["numpy.int64"]
    return _load_skops(POP_RFC_FILEPATH, expected_types)


def _load_superpop_rfc() -> RandomForestClassifier:
    """Load RFC from disk."""
    expected_types = ["numpy.int64"]
    return _load_skops(SUPERPOP_RFC_FILEPATH, expected_types)


def _load_pca() -> RandomForestClassifier:
    """Load RFC from disk."""
    expected_types = [
        "allel.stats.decomposition.GenotypePCA",
        "allel.stats.preprocessing.PattersonScaler",
    ]
    return _load_skops(PCA_FILEPATH, expected_types)


def _load_text(filepath: Path) -> list[str]:
    with filepath.open() as f:
        return [line.strip() for line in f.readlines()]


pop_rfc = _load_pop_rfc()
superpop_rfc = _load_superpop_rfc()
pca = _load_pca()
variants = _load_text(VARIANTS_FILEPATH)
populations = _load_text(POPULATIONS_FILEPATH)
superpops = _load_text(SUPERPOPS_FILEPATH)


def _load_vcf(vcf_filepath: Path, expected_variants: np.ndarray) -> np.ndarray:
    """Load vcf dataframe, restricting to expected variants and marking missing variants as NA."""
    callset = allel.read_vcf(str(vcf_filepath), log=sys.stdout)
    actual_variants = get_variant_ids_from_callset(callset)
    missing_variants = set(expected_variants) - set(actual_variants)
    missing_fraction = len(missing_variants) / len(expected_variants)
    if missing_fraction > MISSING_FRACTION_THRESHOLD:
        logger.warning(
            "VCF: %s missing %s%% of variants",
            vcf_filepath,
            round(missing_fraction * 100, 3),
        )
    samples = callset["samples"]
    genotypes = allel.GenotypeArray(callset["calldata/GT"]).to_n_alt().T
    genotype_df = pd.DataFrame(genotypes, index=samples, columns=actual_variants)
    genotype_df[missing_variants] = pd.NA
    genotype_df = genotype_df[expected_variants]
    return genotype_df


def _infer_ancestry(genotypes: pd.DataFrame) -> pd.DataFrame:
    """Do population- and superpop-level ancestry prediction."""
    genotypes_pc = pca.transform(genotypes)
    predicted_pop_ancestries = pd.DataFrame(
        pop_rfc.predict_proba(genotypes_pc),
        index=genotypes.index,
        columns=populations,
    )
    predicted_superpop_ancestries = pd.DataFrame(
        superpop_rfc.predict_proba(genotypes_pc),
        index=genotypes.index,
        columns=superpops,
    )
    combined_ancestries = predicted_pop_ancestries.join(predicted_superpop_ancestries)
    assert_equal(
        "samples in input",
        genotypes.index,
        "samples in output",
        combined_ancestries.index,
    )
    assert_true("no missing data in output", combined_ancestries.notna().all().all())
    return combined_ancestries


def _get_vcf_from_s3(vcf_s3_path: str) -> str:
    raise NotImplementedError


def predict(vcf_s3_path: str) -> str:
    """Load vcf from s3 path and return predicted ancestry as JSON."""
    vcf = _get_vcf_from_s3(vcf_s3_path)
    genotypes = _load_vcf(vcf, variants)
    inferred_ancestry = _infer_ancestry(genotypes)
    return inferred_ancestry.to_json()
