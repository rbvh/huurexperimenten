"""Typed output schema for the rental-contract extraction demo."""

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects fields not declared in the extraction schema."""

    model_config = ConfigDict(extra="forbid")


class TextEvidence(StrictModel):
    """A text value and the source passage supporting it."""

    value: str | None = Field(
        description="The extracted value, or null when it is not explicitly stated."
    )
    context: str | None = Field(
        description=(
            "A short, verbatim, contiguous excerpt from the supplied Markdown that "
            "contains the label and value, or null when value is null."
        )
    )


class NumberEvidence(StrictModel):
    """A numeric value and the source passage supporting it."""

    value: float | None = Field(
        description="The extracted numeric value, or null when it is not stated."
    )
    context: str | None = Field(
        description=(
            "A short, verbatim, contiguous excerpt from the supplied Markdown that "
            "contains the label and value, or null when value is null."
        )
    )


class PropertyAddress(StrictModel):
    street_address: TextEvidence
    postal_code: TextEvidence
    city: TextEvidence
    country: TextEvidence


class ContractExtraction(StrictModel):
    renter_name: TextEvidence
    total_rent_price: NumberEvidence
    property_address: PropertyAddress
