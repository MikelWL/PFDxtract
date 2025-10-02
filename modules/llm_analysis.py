"""
LLM-based Theme Analysis Module

This module provides an Ollama LLM-based implementation that is interface-compatible
with the existing BERT-based ThemeAnalyzer for seamless integration.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import streamlit as st
import json
import re

# Ollama import for LLM functionality
import ollama

class LLMThemeAnalyzer:
    """
    LLM-based theme analyzer that provides identical interface to BERT ThemeAnalyzer.

    This class maintains full compatibility with the existing Streamlit UI and output
    formats while using Ollama LLM for analysis instead of BERT embeddings.
    """

    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the LLM-based theme analyzer.

        Args:
            model_name: Name of the Ollama model to use for analysis
        """
        self.model_name = model_name

        # Configuration settings (identical to BERT version)
        self.config = {
            "base_similarity_threshold": 0.65,
            "keyword_match_weight": 0.3,
            "semantic_similarity_weight": 0.7,
            "max_themes_per_framework": 5,
            "context_window_size": 200,
        }

        # Initialize frameworks (copied from BERT version)
        self.frameworks = {
            "I-SIRch": self._get_isirch_framework(),
            "House of Commons": self._get_house_of_commons_themes(),
            "Extended Analysis": self._get_extended_themes(),
            "Yorkshire Contributory": self._get_yorkshire_framework(),
        }

        # Theme color mapping (identical to BERT version)
        self.theme_color_map = {}
        self.theme_colors = [
            "#FF5733", "#33FF57", "#3357FF", "#FF33F5", "#F5FF33",
            "#33F5FF", "#FF5733", "#5733FF", "#33FF5A", "#FF3357",
            "#57FF33", "#3357FF", "#FF5A33", "#5AFF33", "#335AFF",
            "#FF335A", "#33FF5A", "#5A33FF", "#FF5A57", "#57FF5A"
        ]

        # Initialize Ollama client (placeholder for now)
        self.llm_client = None
        self._initialize_llm_client()

        logging.info(f"LLMThemeAnalyzer initialized with model: {model_name}")

    def _initialize_llm_client(self):
        """Initialize the Ollama client with real API connection."""
        try:
            # Initialize Ollama client
            self.llm_client = ollama.Client()

            # Test connection by listing available models
            models_response = self.llm_client.list()
            available_models = [model.model for model in models_response.models]

            # Validate that our selected model is available (handle :latest suffix)
            exact_match = self.model_name in available_models
            partial_match = any(self.model_name in model for model in available_models)

            if not exact_match and not partial_match:
                logging.warning(f"Model '{self.model_name}' not found. Available models: {available_models}")

                # Try to use first available model as fallback
                if available_models:
                    self.model_name = available_models[0]
                    logging.info(f"Using fallback model: {self.model_name}")
                else:
                    raise Exception("No models available in Ollama")
            elif not exact_match and partial_match:
                # Find the full model name with suffix
                for model in available_models:
                    if self.model_name in model:
                        self.model_name = model
                        logging.info(f"Using full model name: {self.model_name}")
                        break

            logging.info(f"LLM client initialized successfully with model: {self.model_name}")
            logging.info(f"Available models: {available_models}")

        except Exception as e:
            logging.error(f"LLM client initialization failed: {e}")
            self.llm_client = None

    def create_detailed_results(self, data: pd.DataFrame, content_column: str = "Content") -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze multiple documents and create detailed results.

        This method maintains identical interface to BERT version for seamless integration.

        Args:
            data: DataFrame containing documents to analyze
            content_column: Name of column containing text content

        Returns:
            Tuple of (results_dataframe, highlighted_texts_dict)
        """
        st.info(f"🔄 LLM Analysis starting for {len(data)} documents using {self.model_name}")

        results = []
        highlighted_texts = {}

        # Progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for idx, (i, row) in enumerate(data.iterrows()):
                # Update progress
                progress = (idx + 1) / len(data)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing document {idx + 1}/{len(data)}: {row.get('Title', 'Unknown')[:50]}...")

                # Extract text content
                text = str(row.get(content_column, ""))
                if not text.strip():
                    continue

                # Generate record ID using pandas index (same as BERT)
                record_id = i

                # Perform LLM-based analysis
                framework_themes, theme_highlights = self._analyze_document_llm(text, record_id)

                # Convert results to standardized format
                for framework, themes in framework_themes.items():
                    for theme_name, theme_data in themes.items():
                        results.append({
                            "Record ID": record_id,
                            "Title": row.get("Title", ""),
                            "Framework": framework,
                            "Theme": theme_name,
                            "Confidence": self._get_confidence_label(theme_data["score"]),
                            "Combined Score": theme_data["score"],
                            "Matched Keywords": ", ".join(theme_data.get("keywords", [])),
                            "Matched Sentences": "; ".join(theme_data.get("sentences", [])),

                            # Preserve metadata columns
                            "coroner_name": row.get("coroner_name", ""),
                            "coroner_area": row.get("coroner_area", ""),
                            "year": row.get("year", ""),
                            "date_of_report": row.get("date_of_report", ""),

                            # Analysis metadata
                            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Model Used": self.model_name,
                            "Framework Version": "LLM-v1.0",
                        })

                # Store highlighted texts
                if record_id in theme_highlights:
                    highlighted_texts[record_id] = theme_highlights[record_id]

            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

            results_df = pd.DataFrame(results)

            st.success(f"✅ LLM Analysis completed! Found {len(results)} theme matches across {len(data)} documents.")

            return results_df, highlighted_texts

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ LLM Analysis failed: {str(e)}")
            logging.error(f"LLM analysis error: {e}", exc_info=True)

            # Return empty results to prevent crashes
            return pd.DataFrame(), {}

    def _analyze_document_llm(self, text: str, record_id: str) -> Tuple[Dict, Dict]:
        """
        Analyze a single document using LLM.

        Args:
            text: Document text to analyze
            record_id: Unique identifier for the document

        Returns:
            Tuple of (framework_themes, theme_highlights)
        """
        if self.llm_client is None:
            return self._fallback_analysis(text, record_id)

        framework_themes = {}
        theme_highlights = {
            record_id: {
                "title": f"Document {record_id}",
                "highlighted_content": text[:500] + "...",
                "themes": {}
            }
        }

        try:
            # Create structured JSON-based prompt with I-SIRch framework
            i_sirch_themes = self.frameworks["I-SIRch"]

            prompt = f"""
You are analyzing a Prevention of Future Deaths (PFD) report to identify safety themes using the I-SIRch framework.

DOCUMENT TEXT:
{text}

AVAILABLE THEMES (look for ALL that apply - be comprehensive):
{self._format_framework_for_prompt(i_sirch_themes)}

ANALYSIS INSTRUCTIONS:
1. Read through the ENTIRE document systematically
2. Look for evidence of ANY of the themes listed above - even subtle indicators
3. Consider both DIRECT mentions and INDIRECT implications
4. Be LIBERAL in theme identification - it's better to find relevant themes than miss them
5. For each theme you identify:
   - Extract the most relevant sentences as evidence
   - Identify specific keywords that indicate this theme
   - Assess confidence based on evidence strength

CONFIDENCE GUIDELINES:
- High: Clear, explicit evidence with multiple supporting sentences
- Medium: Good evidence with some supporting sentences
- Low: Subtle indicators or single mention but still relevant

Return ONLY valid JSON in this exact format:
{{
  "themes_found": [
    {{
      "theme_name": "exact theme name from list above",
      "confidence": "High|Medium|Low",
      "supporting_sentences": ["sentence with evidence 1", "sentence with evidence 2"],
      "keywords": ["relevant keyword 1", "relevant keyword 2"],
      "reasoning": "why this theme applies to this case"
    }}
  ]
}}

CRITICAL:
- Use EXACT theme names from the list above
- Include ALL relevant themes found - aim for comprehensive coverage
- Extract sentences exactly as written in the document
- Look beyond obvious keywords - consider context and implications
- Return ONLY JSON, no additional text
"""

            # Get LLM response with parameters optimized for comprehensive theme detection
            response = self.llm_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Slightly higher temperature for more creative theme detection
                    "top_p": 0.95,      # Higher top_p for more diverse theme identification
                    "repeat_penalty": 1.1  # Encourage finding multiple different themes
                }
            )

            llm_response = response['response']
            logging.info(f"LLM Response for document {record_id}: {llm_response[:200]}...")

            # Parse the structured JSON response (Sprint 2)
            framework_themes = self._parse_structured_response(llm_response, text, record_id)

        except Exception as e:
            logging.error(f"LLM analysis failed for document {record_id}: {e}")
            return self._fallback_analysis(text, record_id)

        return framework_themes, theme_highlights

    def _parse_structured_response(self, llm_response: str, original_text: str, record_id: str) -> Dict:
        """Parse structured JSON LLM response into framework themes (Sprint 2)."""
        framework_themes = {}

        try:
            # Log the full response for debugging
            logging.info(f"Raw LLM response for {record_id}: {llm_response[:500]}...")

            # Try to extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = llm_response.strip()

            # Parse JSON response
            parsed_response = json.loads(json_str)

            if "themes_found" not in parsed_response:
                logging.warning(f"No 'themes_found' key in response for {record_id}")
                return self._fallback_analysis_simple(original_text)

            # Process each theme found
            i_sirch_themes = {}
            valid_theme_names = set(self.frameworks["I-SIRch"].keys())

            for theme_data in parsed_response["themes_found"]:
                theme_name = theme_data.get("theme_name", "")

                # Validate theme name exists in framework
                if theme_name not in valid_theme_names:
                    logging.warning(f"Invalid theme name '{theme_name}' for {record_id}. Valid names: {valid_theme_names}")
                    continue

                # Validate sentences exist in original document
                supporting_sentences = theme_data.get("supporting_sentences", [])
                validated_sentences = self._validate_sentences(supporting_sentences, original_text)

                if not validated_sentences:
                    logging.warning(f"No valid supporting sentences for theme '{theme_name}' in {record_id}")
                    continue

                # Convert confidence to numeric score
                confidence = theme_data.get("confidence", "Medium")
                score = self._confidence_to_score(confidence)

                i_sirch_themes[theme_name] = {
                    "score": score,
                    "keywords": theme_data.get("keywords", []),
                    "sentences": validated_sentences,
                    "reasoning": theme_data.get("reasoning", ""),
                    "confidence_label": confidence
                }

                logging.info(f"Added theme '{theme_name}' with confidence {confidence} ({score}) for {record_id}")

            if i_sirch_themes:
                framework_themes["I-SIRch"] = i_sirch_themes

            logging.info(f"Successfully parsed {len(i_sirch_themes)} themes for {record_id}")

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed for {record_id}: {e}")
            logging.error(f"Response was: {llm_response[:200]}...")
            return self._fallback_analysis_simple(original_text)
        except Exception as e:
            logging.error(f"Structured response parsing failed for {record_id}: {e}")
            return self._fallback_analysis_simple(original_text)

        return framework_themes

    def _format_framework_for_prompt(self, framework_themes: Dict) -> str:
        """Format framework themes for inclusion in LLM prompt."""
        formatted = []
        for theme_name, theme_data in framework_themes.items():
            keywords = ", ".join(theme_data.get("keywords", []))
            formatted.append(f"- {theme_name}: Keywords include {keywords}")
        return "\n".join(formatted)

    def _validate_sentences(self, sentences: list, original_text: str) -> list:
        """Validate that sentences exist in the original document and enhance with context."""
        validated = []
        for sentence in sentences:
            # Clean up sentence for comparison
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 10:  # Skip very short sentences
                continue

            # Check if sentence exists in original text and get enhanced context
            enhanced_sentence = self._extract_sentence_with_context(sentence_clean, original_text)
            if enhanced_sentence:
                validated.append(enhanced_sentence)
            else:
                # Log missing sentences for debugging
                logging.warning(f"Sentence not found in original text: '{sentence_clean[:100]}...'")

        return validated

    def _extract_sentence_with_context(self, sentence: str, text: str) -> str:
        """Extract sentence with surrounding context to match BERT's longer sentences (~500-800 chars)."""
        # Normalize for searching
        sentence_normalized = re.sub(r'\s+', ' ', sentence.strip().lower())
        text_normalized = re.sub(r'\s+', ' ', text.lower())

        # Find the sentence in the text
        sentence_position = text_normalized.find(sentence_normalized)
        if sentence_position == -1:
            # Try partial matching for sentences that might be modified by LLM
            words = sentence_normalized.split()
            if len(words) >= 3:
                # Look for the first few words of the sentence
                partial_match = ' '.join(words[:min(5, len(words))])
                sentence_position = text_normalized.find(partial_match)

        if sentence_position == -1:
            return None

        # Extract context around the found sentence (target: 500-800 characters like BERT)
        context_start = max(0, sentence_position - 200)  # 200 chars before
        context_end = min(len(text), sentence_position + len(sentence_normalized) + 300)  # 300 chars after

        # Get the original text with proper capitalization
        enhanced_context = text[context_start:context_end].strip()

        # Clean up context boundaries (try to end on sentence boundaries)
        if context_end < len(text):
            # Try to extend to next sentence boundary
            next_period = text.find('.', context_end)
            if next_period != -1 and next_period - context_end < 100:
                enhanced_context = text[context_start:next_period + 1].strip()

        # Ensure minimum length matching BERT's average (aim for 500+ characters)
        if len(enhanced_context) < 300 and context_start > 0:
            # Expand context further if too short
            expanded_start = max(0, sentence_position - 400)
            expanded_end = min(len(text), sentence_position + len(sentence_normalized) + 400)
            enhanced_context = text[expanded_start:expanded_end].strip()

        return enhanced_context if len(enhanced_context) > 50 else sentence

    def _sentence_exists_in_text(self, sentence: str, text: str) -> bool:
        """Check if sentence exists in text with flexible matching."""
        # Remove extra whitespace and normalize
        sentence_normalized = re.sub(r'\s+', ' ', sentence.strip().lower())
        text_normalized = re.sub(r'\s+', ' ', text.lower())

        # Direct substring match
        if sentence_normalized in text_normalized:
            return True

        # Try matching significant parts (50%+ of words)
        sentence_words = sentence_normalized.split()
        if len(sentence_words) < 3:
            return False

        # Check if most words appear in sequence
        words_found = 0
        for i, word in enumerate(sentence_words):
            if word in text_normalized:
                words_found += 1

        return words_found >= len(sentence_words) * 0.6  # 60% of words must match

    def _confidence_to_score(self, confidence: str) -> float:
        """Convert LLM confidence label to numeric score matching BERT's methodology."""
        # BERT analysis showed mean=0.861, std=0.019, range 0.823-0.890
        confidence_map = {
            "High": 0.88,      # Match BERT's upper range (0.88-0.90)
            "Medium": 0.86,    # Match BERT's median range (0.85-0.87)
            "Low": 0.83        # Match BERT's lower range (0.82-0.84)
        }
        return confidence_map.get(confidence, 0.86)

    def _fallback_analysis_simple(self, text: str) -> Dict:
        """Simple fallback when JSON parsing fails."""
        framework_themes = {}
        text_lower = text.lower()

        # Basic keyword detection for I-SIRch themes
        if any(word in text_lower for word in ['communication', 'information', 'handover']):
            framework_themes.setdefault("I-SIRch", {})["Team - Communication"] = {
                "score": 0.70,
                "keywords": ["communication"],
                "sentences": ["JSON parsing failed - using fallback analysis"],
                "reasoning": "Fallback keyword detection"
            }

        return framework_themes

    def _extract_relevant_sentences(self, text: str, keywords: list) -> list:
        """Extract sentences containing specific keywords."""
        import re

        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                cleaned = sentence.strip()
                if cleaned and len(cleaned) > 20:  # Filter out very short sentences
                    relevant_sentences.append(cleaned)

        return relevant_sentences[:3]  # Return up to 3 most relevant sentences

    def _fallback_analysis(self, text: str, record_id: str) -> Tuple[Dict, Dict]:
        """Fallback analysis when LLM is not available."""
        framework_themes = {}
        theme_highlights = {
            record_id: {
                "title": f"Document {record_id} (Fallback)",
                "highlighted_content": text[:500] + "...",
                "themes": {}
            }
        }

        # Simple keyword-based fallback
        text_lower = text.lower()

        if "communication" in text_lower or "information" in text_lower:
            framework_themes["I-SIRch"] = {
                "Team - Communication": {
                    "score": 0.70,
                    "keywords": ["communication", "information"],
                    "sentences": ["LLM not available - using fallback analysis"]
                }
            }

        return framework_themes, theme_highlights

    def _get_confidence_label(self, score: float) -> str:
        """Convert numerical score to confidence label (identical to BERT version)."""
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _create_integrated_html_for_pdf(self, results_df: pd.DataFrame, highlighted_texts: Dict) -> str:
        """
        Create HTML report for results (placeholder implementation).

        This maintains interface compatibility with the BERT version.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Theme Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .theme-highlight {{ background-color: yellow; padding: 2px; }}
                .report-section {{ margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>LLM-Based Theme Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Model Used: {self.model_name}</p>
            <p>Total Theme Matches: {len(results_df)}</p>

            <h2>Analysis Results</h2>
            <p>This is a placeholder HTML report. Full implementation will include
            highlighted text with theme annotations.</p>

            <!-- Placeholder for actual report content -->

        </body>
        </html>
        """
        return html_content

    # Framework definitions (copied from BERT version for compatibility)
    def _get_isirch_framework(self):
        """Return I-SIRch framework definition matching BERT's exact theme taxonomy."""
        return {
            "Internal - Availability (e.g., operating theatres)": {
                "keywords": ["availability", "operating theatre", "theatre", "resource", "capacity", "accessible"]
            },
            "Internal - Time of day (e.g., night working or day of the week)": {
                "keywords": ["time", "night", "day", "weekend", "shift", "hours", "timing", "schedule"]
            },
            "Jobs/Task - Care planning": {
                "keywords": ["care planning", "planning", "care plan", "treatment plan", "discharge", "pathway"]
            },
            "Jobs/Task - Risk assessment": {
                "keywords": ["risk assessment", "risk", "assessment", "evaluation", "identify", "hazard"]
            },
            "Organisation - Communication factor - Between staff": {
                "keywords": ["communication", "handover", "information", "briefing", "staff", "team"]
            },
            "Organisation - Communication factor - Between staff and patient (verbal)": {
                "keywords": ["patient communication", "verbal", "explain", "consent", "inform", "discuss"]
            },
            "Organisation - Documentation": {
                "keywords": ["documentation", "record", "recording", "notes", "charting", "written"]
            },
            "Organisation - National and/or local guidance": {
                "keywords": ["guidance", "guideline", "policy", "protocol", "standard", "procedure", "national", "local"]
            },
            "Organisation - Team culture factor (e.g., patient safety culture)": {
                "keywords": ["culture", "safety culture", "team culture", "environment", "attitude", "behavior"]
            },
            "Person - Patient (characteristics and performance)": {
                "keywords": ["patient", "characteristics", "condition", "performance", "behavior", "compliance"]
            }
        }

    def _get_house_of_commons_themes(self):
        """Return House of Commons framework definition."""
        return {
            "Communication": {
                "keywords": ["communication", "information sharing", "coordination", "liaison"]
            },
            "Care Planning and Delivery": {
                "keywords": ["care planning", "treatment", "intervention", "delivery", "management"]
            },
            "Leadership and Governance": {
                "keywords": ["leadership", "governance", "oversight", "management", "supervision"]
            }
        }

    def _get_extended_themes(self):
        """Return Extended Analysis framework definition."""
        return {
            "Procedural and Process Failures": {
                "keywords": ["procedure", "process", "protocol", "workflow", "system failure"]
            },
            "Communication Breakdowns": {
                "keywords": ["communication breakdown", "miscommunication", "information gap"]
            },
            "System and Resource Issues": {
                "keywords": ["resource", "staffing", "capacity", "system", "infrastructure"]
            }
        }

    def _get_yorkshire_framework(self):
        """Return Yorkshire Contributory framework definition."""
        return {
            "Situational - Team Factors": {
                "keywords": ["team", "collaboration", "coordination", "teamwork"]
            },
            "Individual - Competency and Training": {
                "keywords": ["competency", "training", "qualification", "certification"]
            },
            "Organisational - Policy and Standards": {
                "keywords": ["organizational", "policy", "standard", "governance"]
            }
        }