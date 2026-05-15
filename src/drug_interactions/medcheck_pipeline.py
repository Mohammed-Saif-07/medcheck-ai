"""
MedCheck AI - Complete Pipeline
ML (XGBoost) + RAG (NumPy) + FDA API + Groq LLM
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import requests
import logging
from pathlib import Path
from difflib import SequenceMatcher
from itertools import combinations

from xgboost import XGBClassifier

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'

COMMON_DRUGS = {
    'lisinopril', 'metformin', 'atorvastatin', 'simvastatin',
    'omeprazole', 'amlodipine', 'metoprolol', 'losartan',
    'albuterol', 'gabapentin', 'hydrochlorothiazide', 'sertraline',
    'ibuprofen', 'furosemide', 'warfarin', 'aspirin',
    'levothyroxine', 'pantoprazole', 'pravastatin', 'clopidogrel',
    'montelukast', 'rosuvastatin', 'escitalopram', 'amoxicillin',
    'azithromycin', 'carvedilol', 'prednisone', 'fluticasone',
    'tramadol', 'duloxetine', 'citalopram', 'tamsulosin',
    'apixaban', 'rivaroxaban', 'insulin', 'glipizide',
}

COMMON_DRUG_ALIASES = {
    'matmorfin': 'metformin',
    'metmorfin': 'metformin',
    'metforman': 'metformin',
    'metfornin': 'metformin',
    'warfrin': 'warfarin',
    'warfarn': 'warfarin',
    'asprin': 'aspirin',
    'aspirn': 'aspirin',
    'lisinapril': 'lisinopril',
    'lisinipril': 'lisinopril',
    'atorvastin': 'atorvastatin',
    'amlodapin': 'amlodipine',
}

DRUG_CLASS_TERMS = {
    'aspirin': ['non-steroidal anti-inflammatory', 'nonsteroidal anti-inflammatory', 'nsaid', 'aspirin'],
    'ibuprofen': ['non-steroidal anti-inflammatory', 'nonsteroidal anti-inflammatory', 'nsaid', 'ibuprofen'],
    'naproxen': ['non-steroidal anti-inflammatory', 'nonsteroidal anti-inflammatory', 'nsaid', 'naproxen'],
    'diclofenac': ['non-steroidal anti-inflammatory', 'nonsteroidal anti-inflammatory', 'nsaid', 'diclofenac'],
    'celecoxib': ['cox-2', 'non-steroidal anti-inflammatory', 'nsaid', 'celecoxib'],
    'lisinopril': ['lisinopril', 'ace inhibitor', 'ace inhibitors'],
    'warfarin': ['warfarin', 'anticoagulant', 'blood thinning'],
}

WARFARIN_INTERACTING_DRUGS = {
    'amiodarone': 'HIGH',
    'aspirin': 'HIGH',
    'fluconazole': 'HIGH',
    'clopidogrel': 'HIGH',
    'ibuprofen': 'HIGH',
    'naproxen': 'HIGH',
    'celecoxib': 'MODERATE',
    'diclofenac': 'HIGH',
    'sertraline': 'MODERATE',
    'duloxetine': 'MODERATE',
    'citalopram': 'MODERATE',
    'escitalopram': 'MODERATE',
}

def clean_drug_name(name):
    name = re.sub(r'\s*\d+\.?\d*\s*(mg|mcg|ml|g|iu|units?|tablet|tab|cap|capsule)s?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9\s/-]', ' ', name)
    name = re.sub(r'\s+\d+\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip().lower()
    return name if name else name


class DrugNormalizer:
    def __init__(self):
        self.known_drugs = set(COMMON_DRUGS)
        self.known_drugs.update(COMMON_DRUG_ALIASES.values())
        self.known_drugs.update(self._load_model_drug_names())

    def _load_model_drug_names(self):
        names = set()

        fda_path = MODELS_DIR / 'fda_interactions.csv'
        if fda_path.exists():
            try:
                df = pd.read_csv(str(fda_path), usecols=['drug1', 'drug2'])
                names.update(clean_drug_name(d) for d in df['drug1'].dropna().astype(str))
                names.update(clean_drug_name(d) for d in df['drug2'].dropna().astype(str))
            except Exception as e:
                logger.warning(f"Could not load FDA drug vocabulary: {e}")

        metadata_path = MODELS_DIR / 'rag_metadata.pkl'
        if metadata_path.exists():
            try:
                with open(str(metadata_path), 'rb') as f:
                    metadata = pickle.load(f)
                for item in metadata:
                    names.add(clean_drug_name(str(item.get('drug1', ''))))
                    names.add(clean_drug_name(str(item.get('drug2', ''))))
            except Exception as e:
                logger.warning(f"Could not load RAG drug vocabulary: {e}")

        return {name for name in names if name}

    def normalize(self, drug_name):
        cleaned = clean_drug_name(drug_name)
        if not cleaned:
            return {
                'input': drug_name,
                'normalized': '',
                'recognized': False,
                'method': 'empty',
                'confidence': 0.0,
            }

        alias = COMMON_DRUG_ALIASES.get(cleaned)
        if alias:
            return {
                'input': drug_name,
                'normalized': alias,
                'recognized': True,
                'method': 'common_alias',
                'confidence': 0.95,
            }

        if cleaned in self.known_drugs:
            return {
                'input': drug_name,
                'normalized': cleaned,
                'recognized': True,
                'method': 'exact',
                'confidence': 1.0,
            }

        best_name = None
        best_score = 0.0
        for candidate in self.known_drugs:
            if abs(len(candidate) - len(cleaned)) > 4:
                continue
            score = SequenceMatcher(None, cleaned, candidate).ratio()
            if score > best_score:
                best_name = candidate
                best_score = score

        if best_name and best_score >= 0.84:
            return {
                'input': drug_name,
                'normalized': best_name,
                'recognized': True,
                'method': 'fuzzy',
                'confidence': best_score,
            }

        return {
            'input': drug_name,
            'normalized': cleaned,
            'recognized': False,
            'method': 'unresolved',
            'confidence': best_score,
            'closest_match': best_name,
        }


class MLPredictor:
    def __init__(self):
        self.model = XGBClassifier()
        self.model.load_model(str(MODELS_DIR / 'xgboost_model.json'))

        try:
            import joblib
            self.encoder = joblib.load(str(MODELS_DIR / 'drug_label_encoder.pkl'))
            self.features = joblib.load(str(MODELS_DIR / 'feature_names.pkl'))
        except Exception:
            self.encoder = None
            self.features = None

        self.fda_df = None
        fda_path = MODELS_DIR / 'fda_interactions.csv'
        if fda_path.exists():
            self.fda_df = pd.read_csv(str(fda_path))

        logger.info("ML Model loaded!")

    def get_fda_data(self, drug1, drug2):
        if self.fda_df is None:
            return None
        d1, d2 = sorted([drug1.lower(), drug2.lower()])
        result = self.fda_df[
            (self.fda_df['drug1'] == d1) & (self.fda_df['drug2'] == d2)
        ]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def predict(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        fda_data = self.get_fda_data(drug1_clean, drug2_clean)

        if self.encoder is None or self.features is None:
            if fda_data:
                return {
                    'known': True,
                    'severity': fda_data.get('severity', 'MODERATE'),
                    'confidence': fda_data.get('severity_score', 0.5),
                    'source': 'FDA FAERS Database',
                    'fda_data': fda_data
                }
            return {'known': False, 'severity': 'UNKNOWN', 'confidence': 0.0}

        try:
            known_drugs = list(self.encoder.classes_)
            if drug1_clean not in known_drugs or drug2_clean not in known_drugs:
                if fda_data:
                    return {
                        'known': True,
                        'severity': fda_data.get('severity', 'MODERATE'),
                        'confidence': fda_data.get('severity_score', 0.5),
                        'source': 'FDA FAERS Database',
                        'fda_data': fda_data
                    }
                return {'known': False, 'severity': 'UNKNOWN', 'confidence': 0.0}

            d1_enc = self.encoder.transform([drug1_clean])[0]
            d2_enc = self.encoder.transform([drug2_clean])[0]

            total = fda_data['total_reports'] if fda_data else 0
            serious = fda_data['serious_reports'] if fda_data else 0
            deaths = fda_data['death_reports'] if fda_data else 0
            hosp = fda_data['hosp_reports'] if fda_data else 0
            sev_score = fda_data['severity_score'] if fda_data else 0
            death_score = fda_data['death_score'] if fda_data else 0
            hosp_score = fda_data['hosp_score'] if fda_data else 0

            X = pd.DataFrame([[
                d1_enc, d2_enc, total, serious, deaths,
                hosp, sev_score, death_score, hosp_score
            ]], columns=self.features)

            proba = self.model.predict_proba(X)[0]
            label = int(self.model.predict(X)[0])
            severity_map = {0: 'LOW', 1: 'MODERATE', 2: 'HIGH'}

            return {
                'known': True,
                'label': label,
                'severity': severity_map.get(label, 'UNKNOWN'),
                'confidence': float(max(proba)),
                'probabilities': {
                    'LOW': float(proba[0]),
                    'MODERATE': float(proba[1]),
                    'HIGH': float(proba[2])
                },
                'source': 'XGBoost ML Model (trained on FDA FAERS)',
                'fda_data': fda_data
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            if fda_data:
                return {
                    'known': True,
                    'severity': fda_data.get('severity', 'MODERATE'),
                    'confidence': 0.5,
                    'source': 'FDA FAERS Database',
                    'fda_data': fda_data
                }
            return {'known': False, 'severity': 'UNKNOWN', 'confidence': 0.0}


class RAGRetriever:
    def __init__(self):
        with open(str(MODELS_DIR / 'rag_metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        documents_path = MODELS_DIR / 'rag_documents.pkl'
        if documents_path.exists():
            with open(str(documents_path), 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self._build_documents()

        embeddings_path = MODELS_DIR / 'rag_embeddings.npy'
        self.embeddings = np.load(str(embeddings_path)) if embeddings_path.exists() else None
        self.embedder = None
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            except Exception as e:
                logger.warning(f"RAG semantic model unavailable; using exact-pair lookup only: {e}")
        else:
            logger.warning("sentence-transformers is not installed; using exact-pair RAG lookup only")

        logger.info(f"RAG loaded: {len(self.metadata)} documents")

    def _build_documents(self):
        self.documents = []
        for m in self.metadata:
            doc = (
                f"{m['drug1'].title()} and {m['drug2'].title()} interaction. "
                f"Severity: {m['severity']}. "
                f"FDA Reports: {m['total_reports']}. "
                f"Serious: {m['serious_reports']}. "
                f"Reactions: {m.get('top_reactions', 'N/A')}."
            )
            self.documents.append(doc)

        with open(str(MODELS_DIR / 'rag_documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

    def _build_embeddings(self):
        self._build_documents()
        if self.embedder is None:
            raise RuntimeError("Cannot build RAG embeddings without a sentence-transformer model")
        self.embeddings = self.embedder.encode(self.documents)
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        np.save(str(MODELS_DIR / 'rag_embeddings.npy'), self.embeddings)

    def _exact_pair_indices(self, drug1, drug2):
        target = frozenset([clean_drug_name(drug1), clean_drug_name(drug2)])
        matches = []
        for idx, metadata in enumerate(self.metadata):
            metadata_pair = frozenset([
                clean_drug_name(str(metadata.get('drug1', ''))),
                clean_drug_name(str(metadata.get('drug2', ''))),
            ])
            if metadata_pair == target:
                matches.append(idx)
        return matches

    def retrieve(self, drug1, drug2, top_k=3):
        exact_indices = self._exact_pair_indices(drug1, drug2)
        if exact_indices:
            results = []
            for idx in exact_indices[:top_k]:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': 1.0
                })
            return results

        return []

    def validate(self, drug1, drug2):
        evidence = self.retrieve(drug1, drug2, top_k=5)
        if not evidence:
            return {'validated': False, 'confidence': 0.0}
        top_score = evidence[0]['score']
        return {
            'validated': top_score > 0.5,
            'confidence': top_score,
            'evidence_count': len(evidence)
        }


class FDAAPIChecker:
    BASE_URL = "https://api.fda.gov/drug/event.json"

    def _term_query(self, drug_name):
        drug = clean_drug_name(drug_name)
        if ' ' in drug:
            return f'patient.drug.medicinalproduct:"{drug}"'
        return f'patient.drug.medicinalproduct:{drug}'

    def check(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        query = (
            f'{self._term_query(drug1_clean)} '
            f'AND {self._term_query(drug2_clean)}'
        )
        try:
            report_response = requests.get(
                self.BASE_URL,
                params={'search': query, 'limit': 1},
                timeout=15
            )
            if report_response.status_code == 404:
                return {
                    'found': False,
                    'query': query,
                    'status_code': 404,
                    'message': 'openFDA returned no co-reported FAERS cases for this pair.'
                }
            if report_response.status_code != 200:
                return {
                    'found': False,
                    'query': query,
                    'status_code': report_response.status_code,
                    'error': report_response.text[:300],
                }

            report_data = report_response.json()
            report_total = report_data.get('meta', {}).get('results', {}).get('total', 0)
            if report_total < 1:
                return {
                    'found': False,
                    'query': query,
                    'message': 'openFDA returned zero co-reported FAERS cases for this pair.'
                }

            reaction_response = requests.get(
                self.BASE_URL,
                params={
                    'search': query,
                    'count': 'patient.reaction.reactionmeddrapt.exact',
                },
                timeout=15
            )
            reactions = []
            if reaction_response.status_code == 200:
                reactions = reaction_response.json().get('results', [])

            if report_total > 1000:
                severity = 'SIGNAL'
            elif report_total > 100:
                severity = 'MODERATE'
            else:
                severity = 'LOW'

            top_reactions = [
                f"{r['term'].title()} ({r['count']} reports)"
                for r in reactions[:5]
            ]
            return {
                'found': True,
                'total_reports': report_total,
                'severity': severity,
                'top_reactions': top_reactions,
                'query': query,
                'source': 'FDA FAERS (Real-time API)',
                'note': (
                    'FAERS counts are co-reported adverse-event cases, not proof that one drug '
                    'caused an interaction with the other.'
                )
            }
        except Exception as e:
            logger.error(f"FDA API error: {e}")
            return {'found': False, 'query': query, 'error': str(e)}


class FDALabelChecker:
    BASE_URL = "https://api.fda.gov/drug/label.json"
    FIELDS = ['drug_interactions', 'warnings', 'boxed_warning', 'contraindications']

    def _local_label(self, drug_name):
        label_path = BASE_DIR / 'data' / 'raw' / 'fda_labels' / f'{drug_name}_label.json'
        if not label_path.exists():
            return []

        try:
            import json
            with open(str(label_path), 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load local FDA label for {drug_name}: {e}")
            return []

        interactions_text = data.get('interactions_text', '')
        if not interactions_text:
            return []

        return [{
            'openfda': {
                'generic_name': [data.get('generic_name') or drug_name],
                'brand_name': data.get('brand_names') or [drug_name],
            },
            'set_id': f'local-{drug_name}',
            'drug_interactions': [interactions_text],
            'source_url': str(label_path),
            'local_source': True,
        }]

    def _search_label(self, drug_name):
        local_results = self._local_label(drug_name)
        if local_results:
            return local_results

        queries = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
        ]
        for query in queries:
            try:
                response = requests.get(
                    self.BASE_URL,
                    params={'search': query, 'limit': 3},
                    timeout=15
                )
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    if results:
                        return results
            except Exception as e:
                logger.error(f"FDA label query error for {drug_name}: {e}")
        return []

    def _snippet_around(self, text, term, window=450):
        match = re.search(rf'\b{re.escape(term)}\b', text, flags=re.IGNORECASE)
        if not match:
            return ""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        sentence_start = max(text.rfind('.', 0, match.start()), text.rfind('\n', 0, match.start()))
        if sentence_start >= start:
            start = sentence_start + 1
        sentence_end_match = re.search(r'\.(?=\s+[A-Z])', text[match.end():end])
        if sentence_end_match:
            end = match.end() + sentence_end_match.end()
        snippet = re.sub(r'\s+', ' ', text[start:end]).strip()
        return snippet.lstrip(') ;:-')

    def _severity_from_text(self, text):
        text_lower = text.lower()
        high_terms = [
            'major or fatal bleeding',
            'fatal bleeding',
            'increase the risk of bleeding',
            'increased risk of bleeding',
            'contraindicated',
        ]
        if any(term in text_lower for term in high_terms):
            return 'HIGH'
        moderate_terms = [
            'closely monitor',
            'monitor',
            'caution',
            'deterioration of renal function',
            'acute renal failure',
            'may be attenuated',
        ]
        if any(term in text_lower for term in moderate_terms):
            return 'MODERATE'
        return 'UNKNOWN'

    def _class_based_snippet(self, text, source_drug, other_drug):
        source_clean = clean_drug_name(source_drug)
        other_clean = clean_drug_name(other_drug)
        other_terms = DRUG_CLASS_TERMS.get(other_clean, [other_clean])
        source_terms = DRUG_CLASS_TERMS.get(source_clean, [source_clean])

        text_lower = text.lower()
        for other_term in other_terms:
            if other_term.lower() not in text_lower:
                continue
            snippet = self._snippet_around(text, other_term, window=520)
            if not snippet:
                continue
            snippet_lower = snippet.lower()
            if any(term.lower() in snippet_lower for term in source_terms):
                return snippet, other_term

        return "", ""

    def _check_one_direction(self, source_drug, other_drug):
        labels = self._search_label(source_drug)
        other_clean = clean_drug_name(other_drug)

        for label in labels:
            openfda = label.get('openfda', {})
            label_name = (
                openfda.get('brand_name')
                or openfda.get('generic_name')
                or [source_drug]
            )[0]
            set_id = label.get('set_id') or label.get('id')

            for field in self.FIELDS:
                text = ' '.join(label.get(field, []))
                if not text:
                    continue
                snippet = self._snippet_around(text, other_clean)
                if snippet:
                    return {
                        'found': True,
                        'severity': self._severity_from_text(snippet),
                        'source': 'FDA Drug Label',
                        'source_url': label.get('source_url', 'https://open.fda.gov/apis/drug/label/'),
                        'label_drug': label_name,
                        'matched_drug': other_clean,
                        'section': field,
                        'snippet': snippet,
                        'set_id': set_id,
                    }
                class_snippet, matched_term = self._class_based_snippet(text, source_drug, other_clean)
                if class_snippet:
                    return {
                        'found': True,
                        'class_based': True,
                        'severity': self._severity_from_text(class_snippet),
                        'source': 'FDA Drug Label',
                        'source_url': label.get('source_url', 'https://open.fda.gov/apis/drug/label/'),
                        'label_drug': label_name,
                        'matched_drug': other_clean,
                        'matched_term': matched_term,
                        'section': field,
                        'snippet': class_snippet,
                        'set_id': set_id,
                        'note': (
                            f'No exact "{other_clean}" mention was found, but the FDA label matched '
                            f'the drug/class term "{matched_term}".'
                        ),
                    }
                if clean_drug_name(label_name).startswith('warfarin'):
                    if other_clean in WARFARIN_INTERACTING_DRUGS and re.search(rf'\b{re.escape(other_clean)}\b', text, flags=re.IGNORECASE):
                        warfarin_snippet = self._snippet_around(text, other_clean, window=520)
                        return {
                            'found': True,
                            'explicit': True,
                            'warfarin_table_match': True,
                            'severity': WARFARIN_INTERACTING_DRUGS[other_clean],
                            'source': 'FDA Drug Label',
                            'source_url': label.get('source_url', 'https://open.fda.gov/apis/drug/label/'),
                            'label_drug': label_name,
                            'matched_drug': other_clean,
                            'section': field,
                            'snippet': warfarin_snippet,
                            'set_id': set_id,
                            'note': (
                                f'Warfarin FDA label lists {other_clean} in an interaction/CYP or '
                                'bleeding-risk section; this overrides ML and FAERS signals.'
                            ),
                        }
                    monitoring_snippet = self._snippet_around(text, 'More frequent INR monitoring', window=320)
                    if not monitoring_snippet:
                        monitoring_snippet = self._snippet_around(text, 'concurrently used drugs', window=320)
                    if monitoring_snippet:
                        return {
                            'found': False,
                            'general_evidence_found': True,
                            'severity': 'MONITOR',
                            'source': 'FDA Drug Label',
                            'source_url': label.get('source_url', 'https://open.fda.gov/apis/drug/label/'),
                            'label_drug': label_name,
                            'matched_drug': other_clean,
                            'section': field,
                            'snippet': monitoring_snippet,
                            'set_id': set_id,
                            'note': (
                                'No pair-specific FDA label sentence was found, but the warfarin label '
                                'contains general guidance for concurrently used drugs and INR monitoring.'
                            ),
                        }
        return {'found': False}

    def check(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        first = self._check_one_direction(drug1_clean, drug2_clean)
        second = self._check_one_direction(drug2_clean, drug1_clean)

        candidates = [r for r in (first, second) if r.get('found') or r.get('general_evidence_found')]
        if candidates:
            severity_rank = {'HIGH': 4, 'MODERATE': 3, 'LOW': 2, 'UNKNOWN': 1, 'MONITOR': 1}
            candidates.sort(
                key=lambda r: (
                    1 if r.get('found') else 0,
                    1 if r.get('warfarin_table_match') else 0,
                    severity_rank.get(r.get('severity', 'UNKNOWN'), 0),
                ),
                reverse=True
            )
            return candidates[0]

        return {'found': False, 'source': 'FDA Drug Label'}


class RxNormChecker:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    def get_concept(self, drug_name):
        query = clean_drug_name(drug_name)
        if not query:
            return {'found': False, 'input': drug_name}

        try:
            response = requests.get(
                f"{self.BASE_URL}/approximateTerm.json",
                params={'term': query, 'maxEntries': 1},
                timeout=10
            )
            if response.status_code != 200:
                return {'found': False, 'input': drug_name, 'status_code': response.status_code}

            candidates = response.json().get('approximateGroup', {}).get('candidate', [])
            if not candidates:
                return {'found': False, 'input': drug_name}

            candidate = next(
                (
                    c for c in candidates
                    if c.get('name') and c.get('source') == 'RXNORM'
                ),
                next((c for c in candidates if c.get('name')), candidates[0])
            )
            score = float(candidate.get('score', 0))
            if score and score < 60:
                return {
                    'found': False,
                    'input': drug_name,
                    'candidate_name': candidate.get('name', query),
                    'candidate_rxcui': candidate.get('rxcui'),
                    'score': score,
                    'reason': 'low RxNorm approximate match score',
                }

            return {
                'found': True,
                'input': drug_name,
                'name': candidate.get('name', query),
                'rxcui': candidate.get('rxcui'),
                'score': score,
                'source': 'RxNorm approximateTerm'
            }
        except Exception as e:
            logger.error(f"RxNorm concept error for {drug_name}: {e}")
            return {'found': False, 'input': drug_name, 'error': str(e)}

    def get_interactions(self, rxcui):
        if not rxcui:
            return {}

        try:
            response = requests.get(
                f"{self.BASE_URL}/interaction/interaction.json",
                params={'rxcui': rxcui, 'sources': 'ONCHigh'},
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            return {
                'api_unavailable': True,
                'status_code': response.status_code,
                'message': 'RxNorm Drug-Drug Interaction API did not return interaction data.'
            }
        except Exception as e:
            logger.error(f"RxNorm interaction error for RxCUI {rxcui}: {e}")
            return {'api_unavailable': True, 'error': str(e)}

    def _find_pair(self, interactions_data, other_rxcui, other_name):
        if interactions_data.get('api_unavailable'):
            return {
                'found': False,
                'api_unavailable': True,
                'status_code': interactions_data.get('status_code'),
                'error': interactions_data.get('error'),
                'message': interactions_data.get(
                    'message',
                    'RxNorm interaction data was unavailable for this request.'
                ),
            }

        groups = interactions_data.get('interactionTypeGroup', [])
        other_name_clean = clean_drug_name(other_name)

        for group in groups:
            source_name = group.get('sourceName', '')
            for interaction_type in group.get('interactionType', []):
                for pair in interaction_type.get('interactionPair', []):
                    concepts = pair.get('interactionConcept', [])
                    for concept in concepts:
                        item = concept.get('minConceptItem', {})
                        item_rxcui = item.get('rxcui')
                        item_name = item.get('name', '')
                        if item_rxcui == other_rxcui or clean_drug_name(item_name) == other_name_clean:
                            severity = pair.get('severity') or source_name or 'UNKNOWN'
                            severity_upper = severity.upper()
                            if 'HIGH' in severity_upper:
                                severity_upper = 'HIGH'
                            elif 'MODERATE' in severity_upper:
                                severity_upper = 'MODERATE'
                            elif 'LOW' in severity_upper:
                                severity_upper = 'LOW'
                            else:
                                severity_upper = 'UNKNOWN'

                            return {
                                'found': True,
                                'severity': severity_upper,
                                'description': pair.get('description', 'RxNorm reports an interaction for this drug pair.'),
                                'source': 'RxNorm ONCHigh',
                                'source_url': 'https://rxnav.nlm.nih.gov/InteractionAPIs.html',
                                'raw_severity': pair.get('severity'),
                                'source_name': source_name,
                                'interacting_drug_name': item_name
                            }

        return {'found': False}

    def check(self, drug1, drug2):
        concept1 = self.get_concept(drug1)
        concept2 = self.get_concept(drug2)

        result = {
            'found': False,
            'source': 'RxNorm ONCHigh',
            'drug1_concept': concept1,
            'drug2_concept': concept2,
        }

        if not concept1.get('found') or not concept2.get('found'):
            return result

        for primary, other in ((concept1, concept2), (concept2, concept1)):
            interactions = self.get_interactions(primary.get('rxcui'))
            pair = self._find_pair(interactions, other.get('rxcui'), other.get('name', ''))
            if pair.get('found'):
                result.update(pair)
                break
            if pair.get('api_unavailable'):
                result.update(pair)
                break

        result['drug1_normalized'] = concept1.get('name')
        result['drug2_normalized'] = concept2.get('name')
        result['drug1_rxcui'] = concept1.get('rxcui')
        result['drug2_rxcui'] = concept2.get('rxcui')
        return result


class LLMExplainer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.available = bool(self.api_key)

    def explain(self, drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result, clinical_assessment=None):
        if not self.available:
            return self._template_explanation(
                drug1, drug2, rxnorm_result, fda_label_result, ml_result,
                rag_evidence, fda_result, clinical_assessment
            )

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)

            if clinical_assessment:
                return self._groq_clinical_assessment(client, drug1, drug2, clinical_assessment)

            rag_context = ""
            if rag_evidence:
                rag_context = "\n".join([f"- {r['document']}" for r in rag_evidence[:3]])

            fda_info = ""
            if fda_result and fda_result.get('found'):
                fda_info = (
                    f"FDA Reports: {fda_result['total_reports']} adverse events. "
                    f"Top reactions: {', '.join(fda_result.get('top_reactions', [])[:3])}"
                )

            rxnorm_info = ""
            if rxnorm_result and rxnorm_result.get('found'):
                rxnorm_info = (
                    f"RxNorm ONCHigh interaction: {rxnorm_result.get('severity', 'UNKNOWN')}. "
                    f"Description: {rxnorm_result.get('description', 'No description available')}. "
                    f"Normalized drugs: {rxnorm_result.get('drug1_normalized', drug1)} + "
                    f"{rxnorm_result.get('drug2_normalized', drug2)}."
                )

            fda_label_info = ""
            if fda_label_result and fda_label_result.get('found'):
                fda_label_info = (
                    f"FDA label section {fda_label_result.get('section', 'unknown')} "
                    f"for {fda_label_result.get('label_drug', drug1)} indicates "
                    f"{fda_label_result.get('severity', 'UNKNOWN')} risk. "
                    f"Evidence excerpt: {fda_label_result.get('snippet', '')}"
                )
            elif fda_label_result and fda_label_result.get('general_evidence_found'):
                fda_label_info = (
                    f"No pair-specific FDA label sentence was found. General label guidance "
                    f"from {fda_label_result.get('label_drug', drug1)} says: "
                    f"{fda_label_result.get('snippet', '')}"
                )

            prompt = f"""You are a clinical pharmacist explaining drug interactions to a patient.
IMPORTANT: Only use the evidence provided below. Do NOT add information not in the evidence.

Drug 1: {drug1.title()}
Drug 2: {drug2.title()}
ML Prediction: {ml_result.get('severity', 'UNKNOWN')} (confidence: {ml_result.get('confidence', 0):.0%})

FDA Evidence:
{fda_info if fda_info else 'No FDA real-time data available'}

RxNorm Evidence:
{rxnorm_info if rxnorm_info else 'No RxNorm ONCHigh interaction found'}

FDA Label Evidence:
{fda_label_info if fda_label_info else 'No FDA label interaction text found'}

RAG Database Evidence:
{rag_context if rag_context else 'No direct evidence in database'}

Based ONLY on the evidence above, provide:
1. A 2-sentence summary of the interaction risk
2. What the patient should do
3. One warning sign to watch for

Keep it under 120 words. Be clear and simple.
End with a short source line naming only the evidence sources used."""

            response = client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=250,
                temperature=0.1
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._template_explanation(
                drug1, drug2, rxnorm_result, fda_label_result, ml_result,
                rag_evidence, fda_result, clinical_assessment
            )

    def _groq_clinical_assessment(self, client, drug1, drug2, assessment):
        summary = assessment.get('evidence_summary', {})
        prompt = f"""You are a clinical drug-drug interaction reasoning engine.

Use hierarchical evidence weighting:
1. FDA-approved label evidence
2. Established pharmacologic mechanism
3. Trusted clinical references/literature
4. RAG/database evidence
5. ML predictions
6. FAERS co-reporting statistics

Rules:
- Do NOT treat FAERS co-report counts as proof of causality.
- Do NOT let ML alone override lack of clinical evidence.
- Do NOT output percentages or "100% confidence".
- If evidence is weak or indirect, prefer LOW or UNKNOWN.
- Keep the exact output sections below.

Drug Pair: {drug1.title()} + {drug2.title()}
Final Risk Level: {assessment.get('risk_level', 'UNKNOWN')}
Evidence Strength: {assessment.get('evidence_strength', 'WEAK')}
Confidence: {assessment.get('confidence', 'LOW')}

Evidence Summary:
FDA Label Evidence: {summary.get('fda_label', 'None')}
Mechanistic Evidence: {summary.get('mechanistic', 'None')}
Clinical/RAG Evidence: {summary.get('clinical_rag', 'None')}
ML Evidence: {summary.get('ml', 'None')}
FAERS Evidence: {summary.get('faers', 'None')}

Reasoning basis:
{assessment.get('reasoning', '')}

Return:
Drug Pair: <drug1> + <drug2>

Final Risk Level: LOW | MODERATE | HIGH | UNKNOWN

Evidence Summary:
* FDA Label Evidence:
* Mechanistic Evidence:
* Clinical/RAG Evidence:
* ML Evidence:
* FAERS Evidence:

Evidence Strength:
STRONG | MODERATE | WEAK

Confidence:
LOW | MEDIUM | HIGH

Reasoning:
Concise clinical explanation.

Important Caveats:
* FAERS reports are observational and non-causal.
* ML predictions are supportive signals only.
* Absence of evidence is not proof of safety.
* Common co-prescription frequency does not equal interaction.

Final Recommendation:
Conservative clinician-style recommendation."""

        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=550,
            temperature=0.05
        )
        return response.choices[0].message.content

    def _template_explanation(self, drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result, clinical_assessment=None):
        if clinical_assessment:
            summary = clinical_assessment.get('evidence_summary', {})
            return f"""Drug Pair: {drug1.title()} + {drug2.title()}

Final Risk Level: {clinical_assessment.get('risk_level', 'UNKNOWN')}

Evidence Summary:
* FDA Label Evidence: {summary.get('fda_label', 'None')}
* Mechanistic Evidence: {summary.get('mechanistic', 'None')}
* Clinical/RAG Evidence: {summary.get('clinical_rag', 'None')}
* ML Evidence: {summary.get('ml', 'None')}
* FAERS Evidence: {summary.get('faers', 'None')}

Evidence Strength:
{clinical_assessment.get('evidence_strength', 'WEAK')}

Confidence:
{clinical_assessment.get('confidence', 'LOW')}

Reasoning:
{clinical_assessment.get('reasoning', 'The available evidence does not support a more specific conclusion.')}

Important Caveats:
* FAERS reports are observational and non-causal.
* ML predictions are supportive signals only.
* Absence of evidence is not proof of safety.
* Common co-prescription frequency does not equal interaction.

Final Recommendation:
{clinical_assessment.get('recommendation', 'Confirm this medication combination with a clinician or pharmacist, especially if the patient has risk factors or symptoms.')}"""

        if rxnorm_result and rxnorm_result.get('found'):
            explanation = (
                f"**{drug1.title()} + {drug2.title()}** — "
                f"RxNorm ONCHigh reports this as a **{rxnorm_result.get('severity', 'UNKNOWN')}** interaction. "
                f"{rxnorm_result.get('description', '')}"
            )
            explanation += (
                "\n\n**Recommendation:** Contact your doctor or pharmacist before taking these together."
                f"\n\n*Source: {rxnorm_result.get('source', 'RxNorm ONCHigh')}*"
            )
            return explanation

        if fda_label_result and fda_label_result.get('found'):
            explanation = (
                f"**{drug1.title()} + {drug2.title()}** — "
                f"FDA-approved labeling indicates **{fda_label_result.get('severity', 'UNKNOWN')}** interaction risk. "
                f"{fda_label_result.get('snippet', '')}"
            )
            explanation += (
                "\n\n**Recommendation:** Do not treat this as safe without clinician or pharmacist review."
                f"\n\n*Source: FDA Drug Label ({fda_label_result.get('section', 'label section')})*"
            )
            return explanation

        if fda_label_result and fda_label_result.get('general_evidence_found'):
            return (
                f"**{drug1.title()} + {drug2.title()}** — No direct pair-specific interaction "
                "was found in the checked sources. However, the FDA-approved warfarin labeling "
                "does give general guidance for concurrently used drugs: "
                f"{fda_label_result.get('snippet', '')}"
                "\n\n**Recommendation:** Do not treat this as a guarantee of safety. If the patient "
                "takes warfarin, confirm the full medication list with a clinician or pharmacist and "
                "follow INR monitoring instructions."
                f"\n\n*Source: FDA Drug Label ({fda_label_result.get('section', 'label section')})*"
            )

        source = 'available evidence'
        severity = 'UNKNOWN'
        if fda_result and fda_result.get('found'):
            severity = fda_result.get('severity', 'UNKNOWN')
            source = 'FDA FAERS real-time data'
        elif fda_label_result and fda_label_result.get('found'):
            severity = fda_label_result.get('severity', 'UNKNOWN')
            source = 'FDA drug label'
        elif rag_evidence:
            severity = rag_evidence[0].get('metadata', {}).get('severity', 'UNKNOWN')
            source = 'local RAG evidence'
        elif ml_result.get('known'):
            severity = ml_result.get('severity', 'UNKNOWN')
            source = 'the ML model'

        confidence = ml_result.get('confidence', 0)

        explanation = f"**{drug1.title()} + {drug2.title()}** — "

        if severity == 'HIGH':
            explanation += (
                f"This combination has been flagged as **HIGH RISK** based on {source}. "
                "There are significant reports of adverse reactions when these medications are used together."
            )
        elif severity == 'MODERATE':
            explanation += (
                f"This combination has **MODERATE** interaction risk based on {source}. "
                "Some adverse events have been reported with this combination."
            )
        elif severity == 'LOW':
            explanation += (
                f"This combination appears to have **LOW** interaction risk based on {source}. "
                "Few adverse events have been reported."
            )
        else:
            explanation += "The available sources did not provide a clear severity classification for this pair."

        if fda_result and fda_result.get('found'):
            reactions = fda_result.get('top_reactions', [])[:3]
            if reactions:
                explanation += f"\n\n**Reported reactions:** {', '.join(reactions)}"

        explanation += (
            f"\n\n**Recommendation:** Contact your doctor or pharmacist before taking these together."
            f"\n\n*Source: {source} | ML Confidence: {confidence:.0%}*"
        )
        return explanation


class MedCheckPipeline:
    def __init__(self, groq_api_key=None):
        logger.info("Loading MedCheck AI Pipeline...")
        self.normalizer = DrugNormalizer()
        self.ml = MLPredictor()
        self.rag = RAGRetriever()
        self.rxnorm = RxNormChecker()
        self.fda_label = FDALabelChecker()
        self.fda = FDAAPIChecker()
        self.llm = LLMExplainer(api_key=groq_api_key)
        logger.info("Pipeline ready!")

    def _mechanism_for_pair(self, drug1, drug2, fda_label_result):
        if fda_label_result.get('warfarin_table_match'):
            return "Warfarin has narrow therapeutic index; listed interacting drugs may increase bleeding risk or alter anticoagulant effect."
        if fda_label_result.get('class_based'):
            matched_term = fda_label_result.get('matched_term', 'pharmacologic class')
            return (
                f"Class-based pharmacologic mechanism inferred from FDA label: "
                f"{matched_term} with {fda_label_result.get('label_drug', drug1)}."
            )

        classes = {
            drug1: DRUG_CLASS_TERMS.get(clean_drug_name(drug1), []),
            drug2: DRUG_CLASS_TERMS.get(clean_drug_name(drug2), []),
        }
        if any('nsaid' in term for term in classes[drug1]) and any('ace inhibitor' in term for term in classes[drug2]):
            return "NSAIDs may reduce ACE-inhibitor antihypertensive effect and increase renal risk in susceptible patients."
        if any('ace inhibitor' in term for term in classes[drug1]) and any('nsaid' in term for term in classes[drug2]):
            return "NSAIDs may reduce ACE-inhibitor antihypertensive effect and increase renal risk in susceptible patients."
        return "No established mechanism identified from local rules."

    def _clinical_assessment(self, drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result):
        fda_found = fda_label_result.get('found', False)
        fda_class = fda_label_result.get('class_based', False)
        rxnorm_found = rxnorm_result.get('found', False)
        rag_found = bool(rag_evidence)
        ml_known = ml_result.get('known', False)
        faers_found = fda_result.get('found', False)

        fda_summary = "No pair-specific FDA label interaction text found."
        if fda_found:
            match_type = "class-based" if fda_class else "explicit"
            fda_summary = (
                f"{match_type} FDA label evidence from {fda_label_result.get('label_drug', 'label')}: "
                f"{fda_label_result.get('snippet', '')}"
            )
        elif fda_label_result.get('general_evidence_found'):
            fda_summary = (
                f"General label guidance only from {fda_label_result.get('label_drug', 'label')}: "
                f"{fda_label_result.get('snippet', '')}"
            )

        mechanism = self._mechanism_for_pair(drug1, drug2, fda_label_result)
        rag_summary = "No direct local RAG evidence found."
        if rag_found:
            top = rag_evidence[0].get('metadata', {})
            rag_summary = (
                f"Local database contains this pair with severity {top.get('severity', 'UNKNOWN')} "
                f"and {top.get('total_reports', 0)} historical reports."
            )

        ml_summary = "Pair not in ML training vocabulary."
        if ml_known:
            ml_summary = (
                f"ML predicts {ml_result.get('severity', 'UNKNOWN')}; treated as supportive only, "
                "not standalone clinical proof."
            )

        faers_summary = "No live FAERS co-report signal returned."
        if faers_found:
            faers_summary = (
                f"Live FAERS has {fda_result.get('total_reports', 0):,} co-reported cases. "
                "This is observational and non-causal."
            )

        mechanism_found = mechanism != "No established mechanism identified from local rules."
        all_evidence_sources_are_empty = (
            not fda_found
            and not fda_label_result.get('general_evidence_found')
            and not rxnorm_found
            and not mechanism_found
            and not rag_found
        )

        if all_evidence_sources_are_empty:
            risk = 'NONE'
            strength = 'NONE'
            confidence = 'HIGH'
            status = 'NO CLINICALLY SIGNIFICANT INTERACTION DETECTED'
            reasoning = "No FDA label interaction, curated database match, mechanism match, or RAG evidence was found."
            recommendation = "No clinically significant interaction was detected in the checked clinical evidence; continue routine clinical review for patient-specific factors."
            decision_source = 'TRUE_NEGATIVE'
        elif fda_found:
            risk = fda_label_result.get('severity', 'UNKNOWN')
            if risk not in {'LOW', 'MODERATE', 'HIGH'}:
                risk = 'MODERATE'
            status = 'CLINICALLY SIGNIFICANT INTERACTION DETECTED'
            strength = 'MODERATE' if fda_class else 'STRONG'
            confidence = 'MEDIUM' if fda_class else 'HIGH'
            reasoning = (
                "FDA label evidence is the primary driver. "
                + (
                    "The label matched the second drug through pharmacologic class rather than an exact drug-name mention. "
                    if fda_class else
                    "The label directly identifies the interacting drug or risk. "
                )
                + "ML and FAERS are considered supportive only."
            )
            recommendation = (
                "Review renal function, blood pressure control, patient risk factors, and alternatives or monitoring "
                "with a clinician/pharmacist before using this combination routinely."
                if fda_class else
                "Follow FDA label precautions and confirm dosing/monitoring with a clinician or pharmacist."
            )
            decision_source = 'FDA'
        elif fda_label_result.get('general_evidence_found'):
            risk = 'LOW'
            status = 'GENERAL MONITORING GUIDANCE FOUND'
            strength = 'WEAK'
            confidence = 'LOW'
            reasoning = (
                "Only general FDA label guidance was found, not a pair-specific interaction. "
                "This is evidence to review the combination, but not enough to classify a moderate/high DDI."
            )
            recommendation = "Confirm patient-specific monitoring needs with a clinician or pharmacist."
            decision_source = 'FDA_GENERAL'
        elif rxnorm_found:
            risk = rxnorm_result.get('severity', 'UNKNOWN')
            if risk not in {'LOW', 'MODERATE', 'HIGH'}:
                risk = 'MODERATE'
            status = 'CLINICALLY SIGNIFICANT INTERACTION DETECTED'
            strength = 'MODERATE'
            confidence = 'MEDIUM'
            reasoning = "RxNorm reports an interaction, but no stronger FDA label evidence was found in the checked sources."
            recommendation = "Confirm the interaction and monitoring plan with a clinician or pharmacist."
            decision_source = 'DB'
        elif rag_found:
            risk = rag_evidence[0].get('metadata', {}).get('severity', 'UNKNOWN')
            if risk not in {'LOW', 'MODERATE', 'HIGH'}:
                risk = 'LOW'
            status = 'SUPPORTIVE EVIDENCE FOUND'
            strength = 'WEAK'
            confidence = 'LOW'
            reasoning = "Only local database/RAG support was found; use it as lower-priority clinical support."
            recommendation = "Treat as a low-confidence signal and verify with a clinician or stronger clinical reference."
            decision_source = 'RAG'
        elif ml_known and ml_result.get('severity') != 'LOW':
            risk = 'LOW'
            status = 'SUPPORTIVE SIGNAL ONLY'
            strength = 'WEAK'
            confidence = 'LOW'
            reasoning = "ML predicted a possible interaction, but ML alone is not clinical evidence; final risk is kept low."
            recommendation = "Do not label this as a clinically significant interaction without stronger evidence."
            decision_source = 'ML_SUPPORTIVE'
        elif faers_found:
            risk = 'LOW'
            status = 'SUPPORTIVE SIGNAL ONLY'
            strength = 'WEAK'
            confidence = 'LOW'
            reasoning = "FAERS co-reporting is observational and cannot establish causality; final risk is kept low."
            recommendation = "Use FAERS as a signal only; check authoritative references before changing therapy."
            decision_source = 'FAERS_SUPPORTIVE'
        else:
            risk = 'NONE'
            strength = 'NONE'
            confidence = 'HIGH'
            status = 'NO CLINICALLY SIGNIFICANT INTERACTION DETECTED'
            reasoning = "No pair-specific clinical evidence was found in the checked sources."
            recommendation = "No clinically significant interaction was detected in the checked clinical evidence; continue routine clinical review for patient-specific factors."
            decision_source = 'TRUE_NEGATIVE'

        return {
            'status': status,
            'risk_level': risk,
            'evidence_strength': strength,
            'confidence': confidence,
            'evidence_summary': {
                'fda_label': fda_summary,
                'mechanistic': mechanism,
                'clinical_rag': rag_summary,
                'ml': ml_summary,
                'faers': faers_summary,
            },
            'reasoning': reasoning,
            'recommendation': recommendation,
            'decision_source': decision_source,
        }

    def _evidence_sources(self, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result):
        sources = []
        if fda_label_result.get('found') or fda_label_result.get('general_evidence_found'):
            sources.append('FDA')
        if rxnorm_result.get('found'):
            sources.append('DB')
        if rag_evidence:
            sources.append('RAG')
        if fda_result.get('found'):
            sources.append('FAERS')
        if ml_result.get('known'):
            sources.append('ML')
        return sources

    def check_interaction(self, drug1, drug2):
        logger.info(f"Stage C: evaluating pair {drug1} + {drug2}")
        drug1_resolution = self.normalizer.normalize(drug1)
        drug2_resolution = self.normalizer.normalize(drug2)
        drug1_clean = drug1_resolution['normalized']
        drug2_clean = drug2_resolution['normalized']

        result = {
            'sources_used': [],
            'drug_resolution': {
                'drug1': drug1_resolution,
                'drug2': drug2_resolution,
            },
            'input_warnings': [],
        }

        if not drug1_resolution['recognized']:
            result['input_warnings'].append(
                f"Could not verify '{drug1}'. Closest local match: "
                f"{drug1_resolution.get('closest_match', 'none')}."
            )
        if not drug2_resolution['recognized']:
            result['input_warnings'].append(
                f"Could not verify '{drug2}'. Closest local match: "
                f"{drug2_resolution.get('closest_match', 'none')}."
            )

        result['needs_review'] = bool(result['input_warnings'])

        # Step 0: RxNorm real-time clinical interaction check
        rxnorm_result = self.rxnorm.check(drug1_clean, drug2_clean)
        result['rxnorm_interaction'] = rxnorm_result
        result['sources_used'].append('RxNorm ONCHigh API')

        if rxnorm_result.get('drug1_normalized'):
            drug1_clean = clean_drug_name(rxnorm_result['drug1_normalized'])
        if rxnorm_result.get('drug2_normalized'):
            drug2_clean = clean_drug_name(rxnorm_result['drug2_normalized'])

        result.update({
            'drug1_input': drug1,
            'drug2_input': drug2,
            'drug1': drug1_clean.title(),
            'drug2': drug2_clean.title(),
            'drug1_clean': drug1_clean,
            'drug2_clean': drug2_clean,
        })

        # Step 1: FDA label interaction evidence
        fda_label_result = self.fda_label.check(drug1_clean, drug2_clean)
        result['fda_label'] = fda_label_result
        result['sources_used'].append('FDA Drug Label API')

        # Step 2: ML Prediction
        ml_result = self.ml.predict(drug1_clean, drug2_clean)
        result['ml_prediction'] = ml_result
        result['sources_used'].append('XGBoost ML Model')

        # Step 3: RAG Evidence
        rag_evidence = self.rag.retrieve(drug1_clean, drug2_clean)
        result['rag_evidence'] = rag_evidence
        result['sources_used'].append('RAG Vector Database')

        # Step 4: FDA FAERS Real-time API
        fda_result = self.fda.check(drug1_clean, drug2_clean)
        result['fda_realtime'] = fda_result
        result['sources_used'].append('FDA FAERS API')

        # Step 5: Clinical evidence-weighted classification
        clinical_assessment = self._clinical_assessment(
            drug1_clean, drug2_clean, rxnorm_result, fda_label_result,
            ml_result, rag_evidence, fda_result
        )
        result['clinical_assessment'] = clinical_assessment
        evidence_sources = self._evidence_sources(
            rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result
        )
        result['structured_output'] = {
            'drug_a': drug1_clean,
            'drug_b': drug2_clean,
            'status': clinical_assessment.get('status'),
            'risk_level': clinical_assessment.get('risk_level', 'UNKNOWN'),
            'confidence': clinical_assessment.get('confidence', 'LOW'),
            'evidence_sources': evidence_sources,
            'explanation': clinical_assessment.get('reasoning', ''),
        }
        logger.info(
            "Final decision for %s + %s: %s, confidence=%s, triggered_by=%s",
            drug1_clean,
            drug2_clean,
            clinical_assessment.get('risk_level'),
            clinical_assessment.get('confidence'),
            clinical_assessment.get('decision_source'),
        )
        result['interaction_found'] = (
            clinical_assessment.get('risk_level') in {'MODERATE', 'HIGH'}
            and not result['needs_review']
        )
        result['overall_severity'] = (
            'UNVERIFIED'
            if result['needs_review']
            else clinical_assessment.get('risk_level', 'UNKNOWN')
        )

        # Step 6: LLM Explanation
        explanation = self.llm.explain(
            drug1_clean, drug2_clean, rxnorm_result, fda_label_result,
            ml_result, rag_evidence, fda_result, clinical_assessment
        )
        result['explanation'] = explanation

        # Step 7: RAG Validation
        validation = self.rag.validate(drug1_clean, drug2_clean)
        result['validation'] = validation
        result['sources_used'].append('RAG Validation')

        if self.llm.available:
            result['sources_used'].append('Groq LLM (Llama 3.1)')

        return result

    def _normalize_drug_list(self, drugs):
        normalized = []
        seen = set()
        for drug in drugs:
            resolution = self.normalizer.normalize(drug)
            canonical = resolution.get('normalized') or clean_drug_name(drug)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            normalized.append({
                'input': drug,
                'normalized': canonical,
                'display': canonical.title(),
                'resolution': resolution,
            })
        logger.info("Stage A: normalized %d input drugs to %d unique drugs", len(drugs), len(normalized))
        return normalized

    def _generate_pairs(self, normalized_drugs):
        pairs = list(combinations(normalized_drugs, 2))
        logger.info("Stage B: generated %d unique unordered drug pairs", len(pairs))
        return pairs

    def _validate_pair_coverage(self, normalized_drugs, results):
        expected_pairs = len(normalized_drugs) * (len(normalized_drugs) - 1) // 2
        evaluated_pairs = {
            frozenset([r.get('drug1_clean'), r.get('drug2_clean')])
            for r in results
        }
        expected_pair_keys = {
            frozenset([a['normalized'], b['normalized']])
            for a, b in combinations(normalized_drugs, 2)
        }
        missing_pairs = expected_pair_keys - evaluated_pairs
        logger.info("Pair validation: expected=%d evaluated=%d missing=%d", expected_pairs, len(results), len(missing_pairs))
        if len(results) != expected_pairs or missing_pairs:
            logger.error("Missing pair evaluations: %s", [sorted(p) for p in missing_pairs])
            raise Exception("ERROR: missing drug pair evaluations")

    def check_multiple(self, drugs):
        normalized_drugs = self._normalize_drug_list(drugs)
        pairs = self._generate_pairs(normalized_drugs)

        results = {
            'drugs': [d['display'] for d in normalized_drugs],
            'normalized_drugs': normalized_drugs,
            'pairs_checked': 0,
            'interactions': [],
            'safe_pairs': [],
            'all_results': [],
            'expected_pairs': len(normalized_drugs) * (len(normalized_drugs) - 1) // 2,
        }

        for drug_a, drug_b in pairs:
            r = self.check_interaction(drug_a['normalized'], drug_b['normalized'])
            results['pairs_checked'] += 1
            results['all_results'].append(r)
            if r['interaction_found']:
                results['interactions'].append(r)
            else:
                results['safe_pairs'].append(r)

        self._validate_pair_coverage(normalized_drugs, results['all_results'])

        return results
