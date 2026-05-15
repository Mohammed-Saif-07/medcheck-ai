"""
Clinical Safety Net
Known critical drug interactions from FDA/WHO guidelines
This is NOT the primary system - it's a BACKUP safety net
for interactions the ML model might miss due to limited training data.

Real hospitals use the same approach:
- Primary: ML/Database lookup
- Safety Net: Curated critical interactions list
"""

CRITICAL_INTERACTIONS = {
    frozenset(['warfarin', 'aspirin']): {
        'severity': 'HIGH',
        'description': 'Significantly increased risk of bleeding. Both drugs affect blood clotting through different mechanisms.',
        'mechanism': 'Warfarin inhibits vitamin K-dependent clotting factors. Aspirin inhibits platelet aggregation. Combined effect greatly increases bleeding risk.',
        'reactions': 'Gastrointestinal bleeding, intracranial hemorrhage, prolonged bleeding time',
        'source': 'FDA Drug Safety Communication',
        'action': 'Avoid combination unless specifically directed by physician. Monitor INR closely if both prescribed.'
    },
    frozenset(['warfarin', 'ibuprofen']): {
        'severity': 'HIGH',
        'description': 'NSAIDs increase bleeding risk when combined with warfarin.',
        'mechanism': 'Ibuprofen inhibits COX enzymes and platelet function. Warfarin is an anticoagulant. Combined use significantly increases bleeding risk.',
        'reactions': 'GI bleeding, bruising, prolonged bleeding',
        'source': 'FDA Safety Alert',
        'action': 'Avoid combination. Use acetaminophen for pain instead.'
    },
    frozenset(['warfarin', 'naproxen']): {
        'severity': 'HIGH',
        'description': 'NSAID-warfarin combination increases bleeding risk.',
        'mechanism': 'Dual antiplatelet and anticoagulant effect',
        'reactions': 'GI bleeding, hemorrhage',
        'source': 'WHO Essential Medicines Guidelines',
        'action': 'Avoid combination.'
    },
    frozenset(['simvastatin', 'clarithromycin']): {
        'severity': 'HIGH',
        'description': 'Risk of severe muscle damage (rhabdomyolysis).',
        'mechanism': 'Clarithromycin inhibits CYP3A4, dramatically increasing simvastatin blood levels.',
        'reactions': 'Rhabdomyolysis, myopathy, kidney failure',
        'source': 'FDA Warning 2011',
        'action': 'Contraindicated. Stop simvastatin during clarithromycin treatment.'
    },
    frozenset(['lisinopril', 'spironolactone']): {
        'severity': 'HIGH',
        'description': 'Dangerous elevation in potassium levels (hyperkalemia).',
        'mechanism': 'Both drugs increase potassium retention through different mechanisms.',
        'reactions': 'Hyperkalemia, cardiac arrhythmias, cardiac arrest',
        'source': 'American Heart Association Guidelines',
        'action': 'Monitor potassium levels closely. May need dose adjustment.'
    },
    frozenset(['methotrexate', 'ibuprofen']): {
        'severity': 'HIGH',
        'description': 'NSAIDs reduce methotrexate clearance, causing toxic levels.',
        'mechanism': 'Ibuprofen reduces renal clearance of methotrexate.',
        'reactions': 'Bone marrow suppression, liver toxicity, kidney failure',
        'source': 'FDA Clinical Guidelines',
        'action': 'Avoid NSAIDs with high-dose methotrexate.'
    },
    frozenset(['fluoxetine', 'tramadol']): {
        'severity': 'HIGH',
        'description': 'Risk of serotonin syndrome - potentially fatal.',
        'mechanism': 'Both drugs increase serotonin levels.',
        'reactions': 'Serotonin syndrome: agitation, confusion, rapid heart rate, seizures',
        'source': 'FDA Drug Safety Communication',
        'action': 'Avoid combination. Use alternative pain medication.'
    },
    frozenset(['aspirin', 'ibuprofen']): {
        'severity': 'MODERATE',
        'description': 'Ibuprofen may reduce the cardioprotective effects of aspirin. Increased GI bleeding risk.',
        'mechanism': 'Competitive inhibition of COX-1. Ibuprofen blocks aspirin from binding.',
        'reactions': 'Reduced aspirin efficacy, GI bleeding',
        'source': 'FDA Alert September 2006',
        'action': 'Take aspirin at least 30 minutes before ibuprofen.'
    },
    frozenset(['metformin', 'glipizide']): {
        'severity': 'MODERATE',
        'description': 'Increased risk of low blood sugar (hypoglycemia).',
        'mechanism': 'Additive glucose-lowering effect.',
        'reactions': 'Hypoglycemia: dizziness, confusion, sweating, shakiness',
        'source': 'American Diabetes Association',
        'action': 'Monitor blood sugar frequently. Adjust doses as needed.'
    },
    frozenset(['amlodipine', 'simvastatin']): {
        'severity': 'MODERATE',
        'description': 'Amlodipine increases simvastatin levels, raising muscle damage risk.',
        'mechanism': 'CYP3A4 enzyme interaction.',
        'reactions': 'Myopathy, elevated CK levels',
        'source': 'FDA Simvastatin Label Update 2011',
        'action': 'Limit simvastatin dose to 20mg when used with amlodipine.'
    },
    frozenset(['lisinopril', 'potassium']): {
        'severity': 'MODERATE',
        'description': 'Risk of hyperkalemia with ACE inhibitor and potassium supplements.',
        'mechanism': 'ACE inhibitors reduce potassium excretion.',
        'reactions': 'Hyperkalemia, cardiac arrhythmias',
        'source': 'Clinical Pharmacology Guidelines',
        'action': 'Monitor potassium levels. Avoid potassium supplements unless directed.'
    },
    frozenset(['insulin', 'glipizide']): {
        'severity': 'HIGH',
        'description': 'High risk of severe hypoglycemia.',
        'mechanism': 'Additive insulin secretion and glucose-lowering effect.',
        'reactions': 'Severe hypoglycemia, loss of consciousness, seizures',
        'source': 'Endocrine Society Guidelines',
        'action': 'Close monitoring required. Adjust doses carefully.'
    },
    frozenset(['clopidogrel', 'omeprazole']): {
        'severity': 'MODERATE',
        'description': 'Omeprazole reduces the effectiveness of clopidogrel.',
        'mechanism': 'Omeprazole inhibits CYP2C19, which activates clopidogrel.',
        'reactions': 'Reduced antiplatelet effect, increased clotting risk',
        'source': 'FDA Drug Safety Communication 2009',
        'action': 'Use pantoprazole instead of omeprazole.'
    },
    frozenset(['digoxin', 'amiodarone']): {
        'severity': 'HIGH',
        'description': 'Amiodarone increases digoxin levels to potentially toxic range.',
        'mechanism': 'Amiodarone inhibits P-glycoprotein and renal clearance of digoxin.',
        'reactions': 'Digoxin toxicity: nausea, visual changes, fatal arrhythmias',
        'source': 'ACC/AHA Guidelines',
        'action': 'Reduce digoxin dose by 50% when starting amiodarone. Monitor levels.'
    },
}


def check_safety_net(drug1, drug2):
    """Check clinical safety net for known critical interactions"""
    d1 = drug1.strip().lower()
    d2 = drug2.strip().lower()

    pair = frozenset([d1, d2])

    if pair in CRITICAL_INTERACTIONS:
        data = CRITICAL_INTERACTIONS[pair]
        return {
            'found': True,
            'severity': data['severity'],
            'description': data['description'],
            'mechanism': data['mechanism'],
            'reactions': data['reactions'],
            'source': data['source'],
            'action': data['action'],
            'type': 'Clinical Safety Net (FDA/WHO Guidelines)'
        }

    return {'found': False}
