"""
MedCheck LLM - Natural Language Explanations

Converts technical medical data into patient-friendly explanations
Uses Claude API or OpenAI GPT (FREE tier!)

Makes complex medical info easy to understand!
"""

import os
import json
from typing import Dict, List
import requests

class LLMExplainer:
    """
    LLM-powered natural language explainer
    Converts medical jargon → simple English
    """
    
    def __init__(self, api_key: str = None, provider: str = "anthropic"):
        """
        Initialize LLM explainer
        
        Args:
            api_key: API key (optional - will check env var)
            provider: "anthropic" (Claude) or "openai" (GPT)
        """
        self.provider = provider
        
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            if provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Check if we have an API key
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            print("   ⚠️  No LLM API key found")
            print("   ℹ️  Will use template explanations")
        else:
            print(f"   ✅ LLM enabled ({provider})")
    
    def explain_interaction(self, drug1: str, drug2: str, 
                           severity: str, evidence: Dict = None) -> str:
        """
        Generate patient-friendly explanation of drug interaction
        """
        
        if not self.enabled:
            # Fallback to template
            return self._template_explanation(drug1, drug2, severity)
        
        # Build prompt for LLM
        prompt = self._build_prompt(drug1, drug2, severity, evidence)
        
        # Call LLM
        try:
            if self.provider == "anthropic":
                explanation = self._call_claude(prompt)
            else:
                explanation = self._call_openai(prompt)
            
            return explanation
        
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            print("   ℹ️  Using template explanation")
            return self._template_explanation(drug1, drug2, severity)
    
    def _build_prompt(self, drug1: str, drug2: str, 
                     severity: str, evidence: Dict = None) -> str:
        """
        Build prompt for LLM
        """
        
        prompt = f"""You are a medical communication expert. Explain this drug interaction in simple, patient-friendly language.

DRUG INTERACTION:
- Drug 1: {drug1}
- Drug 2: {drug2}
- Severity: {severity}

"""
        
        # Add evidence if available
        if evidence and evidence.get('evidence_found'):
            prompt += "\nEVIDENCE FROM FDA:\n"
            for source in evidence.get('sources', [])[:2]:
                prompt += f"- {source.get('section')}: {source.get('text')[:200]}...\n"
            prompt += "\n"
        
        prompt += """
INSTRUCTIONS:
1. Explain in 2-3 simple sentences (6th grade reading level)
2. Avoid medical jargon - use everyday words
3. Focus on WHAT can happen and WHY
4. End with clear action (talk to doctor, monitor symptoms, etc.)
5. Be reassuring but honest

EXAMPLE STYLE:
"These medications both thin your blood. When taken together, they can cause dangerous bleeding like severe nosebleeds or bruising. Please talk to your doctor - they can suggest safer alternatives or adjust your doses."

YOUR EXPLANATION:"""
        
        return prompt
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",  # Fastest, cheapest
            "max_tokens": 300,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['content'][0]['text'].strip()
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",  # Cheapest
            "messages": [
                {"role": "system", "content": "You are a medical communication expert who explains drug interactions in simple language."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    def _template_explanation(self, drug1: str, drug2: str, severity: str) -> str:
        """
        Fallback template explanation (when no API key)
        """
        
        templates = {
            'HIGH': f"""
⚠️ IMPORTANT: {drug1.upper()} and {drug2.upper()} may interact dangerously.

These medications can affect each other in ways that may cause serious side effects. Taking them together could increase risks of bleeding, organ damage, or reduce how well they work.

📞 ACTION REQUIRED: Contact your doctor or pharmacist before taking these medications together. They can:
- Review your complete medication list
- Suggest safer alternatives
- Adjust doses if both are necessary
- Monitor you more closely

Don't stop taking prescribed medications without medical advice.
""",
            'MODERATE': f"""
⚠️ CAUTION: {drug1.upper()} and {drug2.upper()} may interact.

These medications might affect each other, though the risk is moderate. You may need dose adjustments or extra monitoring to use them safely together.

💊 RECOMMENDED: Mention this combination to your doctor at your next visit. They can:
- Confirm it's safe for your situation
- Adjust timing (take at different times of day)
- Monitor for side effects
- Consider alternatives if needed

Continue your medications as prescribed unless your doctor says otherwise.
""",
            'LOW': f"""
ℹ️ AWARENESS: {drug1.upper()} and {drug2.upper()} have a potential interaction.

These medications may interact, but the risk is generally low. Most people can take them together safely with proper monitoring.

✅ SUGGESTION: Keep your doctor informed about all medications you're taking. Watch for any unusual symptoms and report them during your regular checkups.

Continue taking your medications as prescribed.
"""
        }
        
        return templates.get(severity, templates['MODERATE'])
    
    def explain_safe_combination(self, drug1: str, drug2: str) -> str:
        """
        Explain when drugs are safe together
        """
        
        if not self.enabled:
            return f"""
✅ GOOD NEWS: No dangerous interaction found between {drug1.upper()} and {drug2.upper()}.

Based on available medical data, these medications can generally be taken together safely. However:

• Everyone's body is different
• Your specific health conditions matter
• Other medications you take may affect this

💊 BEST PRACTICE: Always keep your doctor and pharmacist informed about all medications, supplements, and vitamins you're taking.

Continue taking your medications as prescribed.
"""
        
        prompt = f"""Explain in 2-3 simple sentences that {drug1} and {drug2} appear safe to take together based on current medical data, but remind them to keep their doctor informed. Be reassuring but encourage communication with healthcare providers. Use 6th grade reading level."""
        
        try:
            if self.provider == "anthropic":
                return self._call_claude(prompt)
            else:
                return self._call_openai(prompt)
        except:
            return self._template_explanation(drug1, drug2, 'LOW')
    
    def suggest_alternatives(self, drug1: str, drug2: str, severity: str) -> str:
        """
        Suggest alternative approaches (template-based for now)
        """
        
        if severity == 'HIGH':
            return """
💡 POSSIBLE ALTERNATIVES (Discuss with your doctor):

Your doctor might consider:
• Using only one medication if possible
• Switching to a different medication class
• Adjusting doses to minimize interaction
• Timing doses differently (morning vs evening)
• Adding protective medications
• Regular monitoring with blood tests

Note: These are general options. Your doctor will recommend what's best for YOUR specific situation based on your complete medical history.
"""
        else:
            return """
💡 MANAGEMENT OPTIONS (Discuss with your doctor):

Your doctor might suggest:
• Taking medications at different times of day
• Regular monitoring for side effects
• Adjusting doses if needed
• Lifestyle modifications to support treatment

These medications may be safe together with proper management.
"""


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo the LLM explainer"""
    
    print("\n" + "="*70)
    print("💊 MedCheck LLM - Natural Language Explainer Demo")
    print("="*70)
    print()
    
    # Initialize (will use template if no API key)
    explainer = LLMExplainer()
    
    # Example 1: HIGH severity
    print("Example 1: HIGH SEVERITY INTERACTION")
    print("-"*70)
    explanation = explainer.explain_interaction(
        "warfarin", "aspirin", "HIGH"
    )
    print(explanation)
    print()
    
    # Example 2: Safe combination
    print("\nExample 2: SAFE COMBINATION")
    print("-"*70)
    explanation = explainer.explain_safe_combination(
        "lisinopril", "metformin"
    )
    print(explanation)
    print()
    
    # Example 3: Alternatives
    print("\nExample 3: ALTERNATIVES SUGGESTION")
    print("-"*70)
    alternatives = explainer.suggest_alternatives(
        "warfarin", "aspirin", "HIGH"
    )
    print(alternatives)
    print()


if __name__ == "__main__":
    demo()
