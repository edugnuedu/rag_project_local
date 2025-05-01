import os
import openai
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query: str, documents: List[Dict]) -> str:
    """
    Generate an answer using GPT-3.5-turbo based on the query and retrieved documents.
    
    Args:
        query (str): The user's question.
        documents (List[Dict]): List of retrieved documents.
    
    Returns:
        str: The generated answer.
    """
    try:
        # Craft the prompt
        prompt = (
            "Answer the following question based on the provided press releases:\n"
            f"Question: {query}\n"
            "Press releases:\n"
        )
        for i, doc in enumerate(documents, 1):
            # Truncate content to manage token limits (approx. 500 tokens per doc)
            truncated_content = doc['content'][:2000]
            prompt += f"{i}. {truncated_content}\n"
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Extract and return the answer
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Sorry, I couldn't generate an answer at this time."
