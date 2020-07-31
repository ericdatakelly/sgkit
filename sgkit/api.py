from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import xarray as xr

from .utils import check_array_like

T = TypeVar("T", bound="DatasetType", covariant=True)
U = TypeVar("U", bound="DatasetType", covariant=True)


class DatasetType(Protocol):
    @classmethod
    def validate_dataset(
        cls, dataset: "SgkitDataset[DatasetType]"
    ) -> "SgkitDataset[DatasetType]":
        ...


class GenotypeCall(DatasetType):
    @classmethod
    def validate_dataset(
        cls, dataset: "SgkitDataset[DatasetType]"
    ) -> "SgkitDataset[DatasetType]":
        print("Check the genotype call specific properties")
        return dataset


class GenotypeDosage(DatasetType):
    @classmethod
    def validate_dataset(
        cls, dataset: "SgkitDataset[DatasetType]"
    ) -> "SgkitDataset[DatasetType]":
        print("Check the genotype dosage specific properties")
        return dataset


class SgkitDataset(Generic[T]):
    class Names:
        # Please keep this sorted:
        CALL_DOSAGE = "call/dosage"
        CALL_DOSAGE_MASK = "call/dosage_mask"
        CALL_GENOTYPE = "call/genotype"
        CALL_GENOTYPE_MASK = "call/genotype_mask"
        CALL_GENOTYPE_PHASED = "call/genotype_phased"
        CONTIGS = "contigs"
        DIM_ALLELE = "alleles"
        DIM_GENOTYPE = "genotypes"
        DIM_PLOIDY = "ploidy"
        DIM_SAMPLE = "samples"
        DIM_VARIANT = "variants"
        SAMPLE_ID = "sample/id"
        VARIANT_ALLELES = "variant/alleles"
        VARIANT_CONTIG = "variant/contig"
        VARIANT_ID = "variant/id"
        VARIANT_POSITION = "variant/position"

    def __init__(self, xr_dataset: xr.Dataset, type_ev: Type[T]):
        # TODO: decide what to check given the type
        if isinstance(type_ev, GenotypeCall):
            # do genotype call checks
            print("I'm validating GenotypeCall dataset")
        self.type_ev = type_ev
        self.xr_dataset = xr_dataset

    # NOTE: this is essentially a map function
    @overload
    def with_dataset(
        self, fun: "Callable[[xr.Dataset], xr.Dataset]", type_ev: Type[U],
    ) -> "SgkitDataset[U]":
        ...

    @overload
    def with_dataset(
        self, fun: "Callable[[xr.Dataset], xr.Dataset]",
    ) -> "SgkitDataset[T]":
        ...

    def with_dataset(
        self,
        fun: "Callable[[xr.Dataset], xr.Dataset]",
        type_ev: Optional[Type[U]] = None,
    ) -> "Union[SgkitDataset[U], SgkitDataset[T]]":
        if type_ev is None:
            return SgkitDataset(fun(self.xr_dataset), self.type_ev)

        return SgkitDataset(fun(self.xr_dataset), type_ev)

    @classmethod
    def create_genotype_call_dataset(
        cls,
        *,
        variant_contig_names: List[str],
        variant_contig: Any,
        variant_position: Any,
        variant_alleles: Any,
        sample_id: Any,
        call_genotype: Any,
        call_genotype_phased: Any = None,
        variant_id: Any = None,
    ) -> "SgkitDataset[GenotypeCall]":
        """Create a dataset of genotype calls.

        Parameters
        ----------
        variant_contig_names : list of str
            The contig names.
        variant_contig : array_like, int
            The (index of the) contig for each variant.
        variant_position : array_like, int
            The reference position of the variant.
        variant_alleles : array_like, S1
            The possible alleles for the variant.
        sample_id : array_like, str
            The unique identifier of the sample.
        call_genotype : array_like, int
            Genotype, encoded as allele values (0 for the reference, 1 for
            the first allele, 2 for the second allele), or -1 to indicate a
            missing value.
        call_genotype_phased : array_like, bool, optional
            A flag for each call indicating if it is phased or not. If
            omitted all calls are unphased.
        variant_id: array_like, str, optional
            The unique identifier of the variant.

        Returns
        -------
        xr.Dataset
            The dataset of genotype calls.

        """
        check_array_like(variant_contig, kind="i", ndim=1)
        check_array_like(variant_position, kind="i", ndim=1)
        check_array_like(variant_alleles, kind="S", ndim=2)
        check_array_like(sample_id, kind="U", ndim=1)
        check_array_like(call_genotype, kind="i", ndim=3)
        data_vars: Dict[Hashable, Any] = {
            cls.Names.VARIANT_CONTIG: ([cls.Names.DIM_VARIANT], variant_contig),
            cls.Names.VARIANT_POSITION: ([cls.Names.DIM_VARIANT], variant_position),
            cls.Names.VARIANT_ALLELES: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_ALLELE],
                variant_alleles,
            ),
            cls.Names.SAMPLE_ID: ([cls.Names.DIM_SAMPLE], sample_id),
            cls.Names.CALL_GENOTYPE: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_SAMPLE, cls.Names.DIM_PLOIDY],
                call_genotype,
            ),
            cls.Names.CALL_GENOTYPE_MASK: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_SAMPLE, cls.Names.DIM_PLOIDY],
                call_genotype < 0,
            ),
        }
        if call_genotype_phased is not None:
            check_array_like(call_genotype_phased, kind="b", ndim=2)
            data_vars[cls.Names.CALL_GENOTYPE_PHASED] = (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_SAMPLE],
                call_genotype_phased,
            )
        if variant_id is not None:
            check_array_like(variant_id, kind="U", ndim=1)
            data_vars[cls.Names.VARIANT_ID] = ([cls.Names.DIM_VARIANT], variant_id)
        attrs: Dict[Hashable, Any] = {cls.Names.CONTIGS: variant_contig_names}
        return SgkitDataset[GenotypeCall](
            xr.Dataset(data_vars=data_vars, attrs=attrs), GenotypeCall
        )

    @classmethod
    def create_genotype_dosage_dataset(
        cls,
        *,
        variant_contig_names: List[str],
        variant_contig: Any,
        variant_position: Any,
        variant_alleles: Any,
        sample_id: Any,
        call_dosage: Any,
        variant_id: Any = None,
    ) -> xr.Dataset:
        """Create a dataset of genotype calls.

        Parameters
        ----------
        variant_contig_names : list of str
            The contig names.
        variant_contig : array_like, int
            The (index of the) contig for each variant.
        variant_position : array_like, int
            The reference position of the variant.
        variant_alleles : array_like, S1
            The possible alleles for the variant.
        sample_id : array_like, str
            The unique identifier of the sample.
        call_dosage : array_like, float
            Dosages, encoded as floats, with NaN indicating a
            missing value.
        variant_id: array_like, str, optional
            The unique identifier of the variant.

        Returns
        -------
        xr.Dataset
            The dataset of genotype calls.

        """
        check_array_like(variant_contig, kind="i", ndim=1)
        check_array_like(variant_position, kind="i", ndim=1)
        check_array_like(variant_alleles, kind="S", ndim=2)
        check_array_like(sample_id, kind="U", ndim=1)
        check_array_like(call_dosage, kind="f", ndim=2)
        data_vars: Dict[Hashable, Any] = {
            cls.Names.VARIANT_CONTIG: ([cls.Names.DIM_VARIANT], variant_contig),
            cls.Names.VARIANT_POSITION: ([cls.Names.DIM_VARIANT], variant_position),
            cls.Names.VARIANT_ALLELES: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_ALLELE],
                variant_alleles,
            ),
            cls.Names.SAMPLE_ID: ([cls.Names.DIM_SAMPLE], sample_id),
            cls.Names.CALL_DOSAGE: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_SAMPLE],
                call_dosage,
            ),
            cls.Names.CALL_DOSAGE_MASK: (
                [cls.Names.DIM_VARIANT, cls.Names.DIM_SAMPLE],
                np.isnan(call_dosage),
            ),
        }
        if variant_id is not None:
            check_array_like(variant_id, kind="U", ndim=1)
            data_vars[cls.Names.VARIANT_ID] = ([cls.Names.DIM_VARIANT], variant_id)
        attrs: Dict[Hashable, Any] = {cls.Names.CONTIGS: variant_contig_names}
        return xr.Dataset(data_vars=data_vars, attrs=attrs)
