# services/integrity_service.py
# Integrity engine service wrapper

from config import Config
from integrity_engine.integrity_engine import IntegrityEngine


def run_integrity_engine():
    """
    Run the integrity engine on saved summaries.
    
    Returns:
        dict: Integrity report with score, risk_level, and reasons
    """
    
    engine = IntegrityEngine(summaries_path=Config.SUMMARIES_DIR)
    engine.load_summaries()
    engine.compute_score()
    report = engine.get_report()
    
    print("\n========== INTEGRITY REPORT ==========")
    print(f"Integrity Score: {report['integrity_score']}")
    print(f"Risk Level: {report['risk_level']}")
    print("Reasons:")
    for reason in report["reasons"]:
        print(f"- {reason}")
    print("======================================")
    
    return report
