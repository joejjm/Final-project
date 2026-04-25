import pandas as pd
import openai
import os

# Set your OpenAI API key (recommended: use environment variable)
# The new OpenAI client will automatically use the OPENAI_API_KEY env variable

# Load the interpreted predictions and features CSV (with NL explanations)
df = pd.read_csv('data/pitch_classifier_predictions_interpreted.csv')


def get_by_label(label):
    return df[df['actual_label'] == label]

def get_examples_for_both_labels(label1, label2, n=3):
    """Retrieve up to n examples for each label."""
    rows1 = df[df['actual_label'] == label1].head(n)
    rows2 = df[df['actual_label'] == label2].head(n)
    return rows1, rows2


def format_context_dual(rows1, rows2, label1, label2):
    context = f"Examples of {label1}s:\n"
    for _, row in rows1.iterrows():
        context += (
            f"Video: {row['video']}, Predicted: {row['pred_combined']}, "
            f"Trajectory Model: {row.get('trajectory_model_nl_explanation', 'No NL explanation')}\n"
        )
    context += f"\nExamples of {label2}s:\n"
    for _, row in rows2.iterrows():
        context += (
            f"Video: {row['video']}, Predicted: {row['pred_combined']}, "
            f"Trajectory Model: {row.get('trajectory_model_nl_explanation', 'No NL explanation')}\n"
        )
    return context

def ask_llm(question, context):
    prompt = f"""
You are a baseball player and expert at reading pitchers. Given the following pitch data:
{context}
If the user asks a question, answer them directly and conversationally, using simple, human-observable cues (for example, "when the glove goes above the cap" or "when the glove is higher than usual"). Do NOT mention technical values, numbers, or measurements (like pixels or raw data). If the user gives a tip, rephrase it as advice to teammates in the dugout, using language like "Hey man, look for..." or "Watch for...". Match your tone to the user's input.
Question or tip:
{question}
"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Retrieve examples for both fastballs and curveballs
    fastball_rows, curveball_rows = get_examples_for_both_labels('fastball', 'curveball', n=3)
    context = format_context_dual(fastball_rows, curveball_rows, 'fastball', 'curveball')
    print("Prompting LLM with context:\n", context)
    first = True
    while True:
        if first:
            question = input("\nI've analyzed the pitcher from the batter's view. How can I help? ")
            first = False
        else:
            question = input("\nAnything else you want to ask? ")
        if question.strip().lower() in ["i'm good, thanks", "im good, thanks", "i am good, thanks", "good, thanks", "done", "exit", "quit"]:
            print("\nSession ended. Good luck at the plate!")
            break
        answer = ask_llm(question, context)
        print("\nLLM Answer:\n", answer)
