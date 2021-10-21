from typing import List
from better_profanity import profanity


def is_contain_profanity_in_triple(triple: List[str]) -> bool:
    return profanity.contains_profanity(" ".join(triple))
