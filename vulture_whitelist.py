# Vulture whitelist - false positives for pydantic V2 @classmethod validators
# These validators require cls parameter even when not used in the method body
cls  # noqa: used in @classmethod pydantic validators (models.py)
