"""MedVLM package for chest X-ray report generation."""

from medvlm.config import MedVLMConfig

__all__ = ["MedVLM", "MedVLMConfig", "ReportTokenizer"]


def __getattr__(name: str):
    if name == "MedVLM":
        from medvlm.model import MedVLM

        return MedVLM
    if name == "ReportTokenizer":
        from medvlm.tokenizer import ReportTokenizer

        return ReportTokenizer
    raise AttributeError(name)
