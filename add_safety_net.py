"""
Run this script to add clinical safety net to the pipeline.
Adds critical interaction checking as a backup to ML model.
"""

import os

pipeline_path = 'src/drug_interactions/medcheck_pipeline.py'

# Read current file
with open(pipeline_path, 'r') as f:
    content = f.read()

# Add import at top
old_import = "from pathlib import Path"
new_import = """from pathlib import Path

try:
    from drug_interactions.clinical_safety_net import check_safety_net
except ImportError:
    from .clinical_safety_net import check_safety_net"""

content = content.replace(old_import, new_import)

# Add safety net check in check_interaction method
old_check = """        # Step 1: ML Prediction
        ml_result = self.ml.predict(drug1, drug2)
        result['ml_prediction'] = ml_result
        result['sources_used'].append('XGBoost ML Model')"""

new_check = """        # Step 0: Clinical Safety Net (critical interactions)
        safety = check_safety_net(drug1_clean, drug2_clean)
        result['safety_net'] = safety
        result['sources_used'].append('Clinical Safety Net')

        # Step 1: ML Prediction
        ml_result = self.ml.predict(drug1, drug2)
        result['ml_prediction'] = ml_result
        result['sources_used'].append('XGBoost ML Model')"""

content = content.replace(old_check, new_check)

# Update interaction detection to include safety net
old_detect = """        has_interaction = (
            ml_result.get('known', False) and ml_result.get('severity', 'LOW') != 'LOW'
        ) or (
            fda_result.get('found', False)
        ) or (
            len(rag_evidence) > 0 and rag_evidence[0]['score'] > 0.6
        )"""

new_detect = """        has_interaction = (
            safety.get('found', False)
        ) or (
            ml_result.get('known', False) and ml_result.get('severity', 'LOW') != 'LOW'
        ) or (
            fda_result.get('found', False)
        ) or (
            len(rag_evidence) > 0 and rag_evidence[0]['score'] > 0.6
        )"""

content = content.replace(old_detect, new_detect)

# Update severity detection to include safety net
old_severity = """        severities = []
        if ml_result.get('known'):
            severities.append(ml_result.get('severity', 'LOW'))
        if fda_result.get('found'):
            severities.append(fda_result.get('severity', 'LOW'))"""

new_severity = """        severities = []
        if safety.get('found'):
            severities.append(safety.get('severity', 'HIGH'))
        if ml_result.get('known'):
            severities.append(ml_result.get('severity', 'LOW'))
        if fda_result.get('found'):
            severities.append(fda_result.get('severity', 'LOW'))"""

content = content.replace(old_severity, new_severity)

# Write updated file
with open(pipeline_path, 'w') as f:
    f.write(content)

print("✅ Pipeline updated with clinical safety net!")
print("✅ Now warfarin + aspirin will show HIGH RISK!")
