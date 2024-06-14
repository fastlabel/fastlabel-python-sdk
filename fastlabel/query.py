from typing import List, Optional, TypedDict


class DatasetObjectGetQuery(TypedDict, total=False):
    dataset: str
    version: str
    revisionId: str
    tags: Optional[List[str]]
    licenses: Optional[List[str]]
    types: Optional[List[str]]
    offset: int
    limit: int
