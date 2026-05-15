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

from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'

def clean_drug_name(name):
    name = re.sub(r'\s*\d+\.?\d*\s*(mg|mcg|ml|g|iu|units?|tablet|tab|cap|capsule)s?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+\d+\s*$', '', name)
    name = name.strip().lower()
    return name if name else name


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
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception as e:
            logger.warning(f"RAG semantic model unavailable; using exact-pair lookup only: {e}")

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

    def check(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        query = (
            f'patient.drug.medicinalproduct:"{drug1_clean}"'
            f'+AND+patient.drug.medicinalproduct:"{drug2_clean}"'
        )
        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    'search': query,
                    'count': 'patient.reaction.reactionmeddrapt.exact',
                    'limit': 10
                },
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    reactions = data['results']
                    total = sum(r['count'] for r in reactions)
                    if total >= 5:
                        if total > 100:
                            severity = 'HIGH'
                        elif total > 30:
                            severity = 'MODERATE'
                        else:
                            severity = 'LOW'

                        top_reactions = [
                            f"{r['term'].title()} ({r['count']} reports)"
                            for r in reactions[:5]
                        ]
                        return {
                            'found': True,
                            'total_reports': total,
                            'severity': severity,
                            'top_reactions': top_reactions,
                            'source': 'FDA FAERS (Real-time API)'
                        }
            return {'found': False}
        except Exception as e:
            logger.error(f"FDA API error: {e}")
            return {'found': False, 'error': str(e)}


class FDALabelChecker:
    BASE_URL = "https://api.fda.gov/drug/label.json"
    FIELDS = ['drug_interactions', 'warnings', 'boxed_warning', 'contraindications']

    def _search_label(self, drug_name):
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
        sentence_end = text.find('.', match.end())
        if sentence_end != -1 and sentence_end <= end:
            end = sentence_end + 1
        return re.sub(r'\s+', ' ', text[start:end]).strip()

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
        if 'closely monitor' in text_lower or 'monitor' in text_lower or 'caution' in text_lower:
            return 'MODERATE'
        return 'UNKNOWN'

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
                        'source_url': 'https://open.fda.gov/apis/drug/label/',
                        'label_drug': label_name,
                        'matched_drug': other_clean,
                        'section': field,
                        'snippet': snippet,
                        'set_id': set_id,
                    }
        return {'found': False}

    def check(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        first = self._check_one_direction(drug1_clean, drug2_clean)
        if first.get('found'):
            return first

        second = self._check_one_direction(drug2_clean, drug1_clean)
        if second.get('found'):
            return second

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
            return {}
        except Exception as e:
            logger.error(f"RxNorm interaction error for RxCUI {rxcui}: {e}")
            return {}

    def _find_pair(self, interactions_data, other_rxcui, other_name):
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

        result['drug1_normalized'] = concept1.get('name')
        result['drug2_normalized'] = concept2.get('name')
        result['drug1_rxcui'] = concept1.get('rxcui')
        result['drug2_rxcui'] = concept2.get('rxcui')
        return result


class LLMExplainer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.available = bool(self.api_key)

    def explain(self, drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result):
        if not self.available:
            return self._template_explanation(drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)

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
            return self._template_explanation(drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result)

    def _template_explanation(self, drug1, drug2, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result):
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
        self.ml = MLPredictor()
        self.rag = RAGRetriever()
        self.rxnorm = RxNormChecker()
        self.fda_label = FDALabelChecker()
        self.fda = FDAAPIChecker()
        self.llm = LLMExplainer(api_key=groq_api_key)
        logger.info("Pipeline ready!")

    def check_interaction(self, drug1, drug2):
        drug1_clean = clean_drug_name(drug1)
        drug2_clean = clean_drug_name(drug2)

        result = {'sources_used': []}

        # Step 0: RxNorm real-time clinical interaction check
        rxnorm_result = self.rxnorm.check(drug1_clean, drug2_clean)
        result['rxnorm_interaction'] = rxnorm_result
        result['sources_used'].append('RxNorm ONCHigh API')

        if rxnorm_result.get('drug1_normalized'):
            drug1_clean = clean_drug_name(rxnorm_result['drug1_normalized'])
        if rxnorm_result.get('drug2_normalized'):
            drug2_clean = clean_drug_name(rxnorm_result['drug2_normalized'])

        result.update({
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

        # Step 4: Determine interaction
        has_interaction = (
            rxnorm_result.get('found', False)
        ) or (
            fda_label_result.get('found', False)
        ) or (
            ml_result.get('known', False) and ml_result.get('severity', 'LOW') != 'LOW'
        ) or (
            fda_result.get('found', False)
        ) or (
            len(rag_evidence) > 0 and rag_evidence[0]['score'] > 0.6
        )

        result['interaction_found'] = has_interaction

        # Determine severity
        severities = []
        if rxnorm_result.get('found'):
            severities.append(rxnorm_result.get('severity', 'UNKNOWN'))
        if fda_label_result.get('found'):
            severities.append(fda_label_result.get('severity', 'UNKNOWN'))
        if ml_result.get('known'):
            severities.append(ml_result.get('severity', 'LOW'))
        if fda_result.get('found'):
            severities.append(fda_result.get('severity', 'LOW'))
        if rag_evidence:
            severities.append(rag_evidence[0].get('metadata', {}).get('severity', 'UNKNOWN'))

        if 'HIGH' in severities:
            result['overall_severity'] = 'HIGH'
        elif 'MODERATE' in severities:
            result['overall_severity'] = 'MODERATE'
        elif 'LOW' in severities:
            result['overall_severity'] = 'LOW'
        elif has_interaction:
            result['overall_severity'] = 'UNKNOWN'
        else:
            result['overall_severity'] = 'NONE'

        # Step 5: LLM Explanation
        explanation = self.llm.explain(
            drug1_clean, drug2_clean, rxnorm_result, fda_label_result, ml_result, rag_evidence, fda_result
        )
        result['explanation'] = explanation

        # Step 6: RAG Validation
        validation = self.rag.validate(drug1_clean, drug2_clean)
        result['validation'] = validation
        result['sources_used'].append('RAG Validation')

        if self.llm.available:
            result['sources_used'].append('Groq LLM (Llama 3.1)')

        return result

    def check_multiple(self, drugs):
        results = {
            'drugs': [d.strip().title() for d in drugs],
            'pairs_checked': 0,
            'interactions': [],
            'safe_pairs': []
        }

        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                r = self.check_interaction(drugs[i], drugs[j])
                results['pairs_checked'] += 1
                if r['interaction_found']:
                    results['interactions'].append(r)
                else:
                    results['safe_pairs'].append(r)

        return results
