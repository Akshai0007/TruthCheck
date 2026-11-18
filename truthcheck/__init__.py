from .env import load_dotenv

load_dotenv()   # picks up TRUTHCHECK_* settings from a local .env file, if there is one

from .llm import make_llm
from .models import ClaimResult, Evidence, Report
from .pipeline import TruthCheck
from .retrieval import LocalDocsRetriever, WikipediaRetriever

__all__ = ["TruthCheck", "make_llm", "LocalDocsRetriever", "WikipediaRetriever",
           "Report", "ClaimResult", "Evidence"]
