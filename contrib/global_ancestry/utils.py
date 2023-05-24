from typing import Any, Dict, List
import numpy as np

def get_variant_ids_from_callset(callset: Dict[str, Any]) -> np.ndarray:
    """Given a callset generated from scikit.allel, return variant ids in Broad notation."""
    # see https://illumina.github.io/NirvanaDocumentation/core-functionality/variant-ids/
    variant_ids = np.array(
        [
            "-".join([chrom, str(pos), ref, alt])
            for chrom, pos, ref, alt in zip(
                callset["variants/CHROM"],
                callset["variants/POS"],
                callset["variants/REF"],
                callset["variants/ALT"][:, 0],
                # matrix contains all alts for position-- we only want the first because we're
                # excluding all multi-allelic variants
                strict=True,
            )
        ],
    )
    return variant_ids
