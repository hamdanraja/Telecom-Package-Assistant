from typing import Literal, Optional
from pydantic import BaseModel, Field

Operator = Literal["jazz", "zong", "ufone"]
Category = Literal["hourly", "daily", "weekly", "fortnightly", "monthly", "long_term"]
SortKey = Literal[
    "price", "data_gb", "price_per_gb", "price_per_day", "validity_days",
    "total_minutes", "sms", "onnet_minutes", "price_per_100_onnet",
]


class Plan(BaseModel):
    """Structured interpretation of the user's question."""
    intent: Literal[
        "package_details", "compare_packages", "compare_operators",
        "recommend", "search", "statistics", "chitchat",
    ] = "search"
    operators: list[Operator] = Field(default_factory=list, description="empty = all operators")
    package_names: list[str] = Field(default_factory=list)
    package_type: Optional[Literal["bundle", "internet", "call", "sms"]] = None
    categories: list[Category] = Field(default_factory=list, description="validity buckets; several allowed")
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_data_gb: Optional[float] = None
    max_data_gb: Optional[float] = None
    min_minutes: Optional[float] = None
    min_sms: Optional[float] = None
    validity_days: Optional[float] = None
    keywords: list[str] = Field(default_factory=list)
    sort_by: Optional[SortKey] = None
    ascending: Optional[bool] = None
    limit: int = 5
    target_data_gb: Optional[float] = None                # centre of a "near X GB" band (set by rules)
    similar_to: list[str] = Field(default_factory=list)   # reference packages for "competitors/similar"