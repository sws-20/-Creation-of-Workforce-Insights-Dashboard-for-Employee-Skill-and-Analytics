import asyncio
import os
import re
from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(BASE_DIR, "docs", "hr_policies.txt")

class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_and_chunk_documents()
        self.build_index()

    def load_and_chunk_documents(self):
        """Loads and splits the HR policy document into sections/chunks."""
        if not os.path.exists(DOCS_PATH):
            # Create default document if not found
            os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
            with open(DOCS_PATH, 'w') as f:
                f.write("Talent Intelligence Corp HR Policy Handbook.\nNo policies defined yet.")
        
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content by sections
        sections = re.split(r'(SECTION \d+: [^\n]+)', content)
        
        # Merge sections with headers
        header = "Talent Intelligence Corp HR Policies & Handbook"
        current_chunk = header
        
        for part in sections:
            part = part.strip()
            if not part:
                continue
            if part.startswith("SECTION "):
                current_chunk = part
            else:
                full_chunk = f"{current_chunk}\n\n{part}"
                self.chunks.append(full_chunk)

    def build_index(self):
        """Builds a TF-IDF retrieval index (fallback or primary)."""
        if not self.chunks:
            return
            
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant chunks for a given query."""
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Only return chunks with non-zero similarity or at least the top chunk
            if score > 0.05 or len(results) == 0:
                results.append({
                    "chunk_id": int(idx),
                    "score": score,
                    "content": self.chunks[idx]
                })
        return results

    def generate_fallback_answer(self, query: str, context_chunks: List[str]) -> str:
        """Rule-based search fallback when no LLM API key is configured."""
        query_lower = query.lower()
        context_text = "\n\n".join(context_chunks)
        
        # Check for common topics in HR policies
        if "training" in query_lower or "skill" in query_lower:
            # Extract key training metrics
            limit_match = re.search(r"Funding Limit: ([^\n.]+)", context_text)
            hours_match = re.search(r"Required Training Hours: ([^\n.]+)", context_text)
            limit_val = limit_match.group(1) if limit_match else "$5,000 annually"
            hours_val = hours_match.group(1) if hours_match else "16 training hours"
            return (
                f"Based on the company HR policy:\n"
                f"- Employees are eligible for professional training with an annual funding limit of {limit_val}.\n"
                f"- A minimum of {hours_val} per calendar year is required for all full-time employees.\n"
                f"- Employees with identified skill gaps are placed on a Performance Improvement Plan (PIP) and must complete 3 courses in the next quarter."
            )
            
        if "work-life" in query_lower or "overtime" in query_lower or "hours" in query_lower or "remote" in query_lower:
            hours_match = re.search(r"Core Working Hours: ([^\n.]+)", context_text)
            ot_match = re.search(r"Overtime Policy: ([^\n.]+)", context_text)
            remote_match = re.search(r"Remote Work: ([^\n.]+)", context_text)
            
            hours_val = hours_match.group(1) if hours_match else "9:00 AM to 5:00 PM, Monday through Friday"
            ot_val = ot_match.group(1) if ot_match else "Pre-approved by manager, standard 80 hours per bi-weekly pay period"
            remote_val = remote_match.group(1) if remote_match else "Eligible for hybrid work (up to 2 days remote per week)"
            
            return (
                f"According to the Work-Life Balance Guidelines:\n"
                f"- Standard core working hours are {hours_val}.\n"
                f"- Overtime policy: {ot_val}. Overtime exceeding 10 hours a week triggers an HR alert.\n"
                f"- Remote work: {remote_val}."
            )
            
        if "promotion" in query_lower or "career" in query_lower or "advance" in query_lower:
            readiness_match = re.search(r"Promotion Readiness Check: \n([\s\S]+?)(?=\n\n|\n-|$)", context_text)
            cycle_match = re.search(r"Review Cycle: ([^\n.]+)", context_text)
            
            readiness_val = readiness_match.group(1) if readiness_match else "- 3 years since last promotion\n- Performance rating >= 3\n- Job Level < 5"
            cycle_val = cycle_match.group(1) if cycle_match else "Bi-annually in June and December"
            
            return (
                f"According to the Career Advancement Policy:\n"
                f"Promotion reviews are conducted {cycle_val}. The eligibility requirements are:\n"
                f"{readiness_val.strip()}"
            )
            
        if "retention" in query_lower or "bonus" in query_lower or "sabbatical" in query_lower or "wellness" in query_lower:
            wellness_match = re.search(r"Wellness Allowance: ([^\n.]+)", context_text)
            sabbatical_match = re.search(r"Sabbatical Leave: ([^\n.]+)", context_text)
            bonus_match = re.search(r"Retention Bonus: ([^\n.]+)", context_text)
            
            wellness_val = wellness_match.group(1) if wellness_match else "$150 per month wellness stipend"
            sabbatical_val = sabbatical_match.group(1) if sabbatical_match else "Eligible for a 4-week paid sabbatical after 7 years"
            bonus_val = bonus_match.group(1) if bonus_match else "Retention bonuses up to 15% for high-risk departments"
            
            return (
                f"Under the Retention and Workforce Wellness Program:\n"
                f"- Wellness Stipend: {wellness_val}.\n"
                f"- Sabbatical: {sabbatical_val}.\n"
                f"- Retention Bonus: {bonus_val} in exchange for a 1-year service commitment."
            )

        # Generic summary fallback
        sentences = context_text.split('.')
        relevant_sentences = []
        for s in sentences:
            if any(word in s.lower() for word in query_lower.split()):
                relevant_sentences.append(s.strip())
                if len(relevant_sentences) >= 3:
                    break
        
        if relevant_sentences:
            return "Based on the retrieved HR documentation:\n" + ". ".join(relevant_sentences) + "."
            
        return "I could not find a specific policy matching your query. Here is the closest section retrieved:\n\n" + context_chunks[0]

    def query(self, user_query: str) -> Dict[str, Any]:
        """Performs full RAG retrieval and generates answers."""
        retrieved = self.retrieve(user_query, top_k=2)
        if not retrieved:
            return {
                "query": user_query,
                "answer": "No relevant HR policies could be found in the system.",
                "context": []
            }
            
        context_contents = [chunk["content"] for chunk in retrieved]
        
        # Check for Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai

                context_str = "\n\n=== CONTEXT ===\n" + "\n\n".join(context_contents)

                async def generate_response():
                    client = genai.Client(api_key=api_key)
                    return await client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"""
{context_str}

Question: {user_query}
""",
                        config={
                            "system_instruction": (
                                "You are a helpful HR Assistant. Answer only from the "
                                "provided HR policy context. If the answer is unavailable, say so."
                            ),
                            "temperature": 0.2,
                            "max_output_tokens": 250,
                        },
                    )

                response = asyncio.run(generate_response())

                answer = response.text.strip()
                source = "Google Gemini model"
            except Exception as e:
                answer = self.generate_fallback_answer(user_query, context_contents)
                source = f"Local Fallback Generator (Gemini failed: {e})"
        else:
            answer = self.generate_fallback_answer(user_query, context_contents)
            source = "Local Fallback Generator (No API Key)"
            
        return {
            "query": user_query,
            "answer": answer,
            "source": source,
            "context": retrieved
        }

if __name__ == "__main__":
    # Test query
    rag = SimpleRAG()
    res = rag.query("What are the requirements for a promotion review?")
    print(f"Query: {res['query']}")
    print(f"Answer:\n{res['answer']}")
