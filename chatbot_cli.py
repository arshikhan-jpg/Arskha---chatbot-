"""
Interactive Command-Line FAQ Chatbot
Provides terminal-based interaction with NLP preprocessing and Cosine Similarity FAQ matching.
"""

import sys
from faq_matcher import FAQMatcher


def run_cli():
    matcher = FAQMatcher()
    print("=" * 65)
    print("🤖 Welcome to Arskha - AI FAQ Assistant (CLI Mode) 🌸")
    print(f"Loaded {len(matcher.faqs)} FAQs across various categories.")
    print("Preprocessed with NLTK & Matched with Cosine Similarity.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'nlp <question>' to see the step-by-step NLP breakdown.")
    print("Type 'help' to see sample topics.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArskha: Goodbye! Have a wonderful day! 🌸💕")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Arskha: Goodbye! Have a wonderful day! 🌸💕")
            break

        if user_input.lower().startswith("nlp "):
            sample = user_input[4:].strip()
            details = matcher.preprocessor.preprocess(sample, return_details=True)
            print("\n--- 🔍 NLP Pipeline Breakdown ---")
            print(f"Raw Input:       {details['raw']}")
            print(f"Cleaned:         {details['cleaned']}")
            print(f"Tokenized:       {details['tokens']}")
            print(f"NoStopwords:     {details['filtered_tokens']}")
            print(f"Lemmatized:      {details['lemmatized_tokens']}")
            print(f"Processed Text:  {details['processed_text']}")
            continue

        result = matcher.find_match(user_input)

        confidence_pct = result.get("score_percent", 0)
        category = result.get("category", "General")
        matched_q = result.get("matched_question")

        print(f"\n🌸 Arskha [{confidence_pct}% Match | {category}]:")
        if matched_q:
            print(f"   (Matched: \"{matched_q}\")")
        print(f"   {result['answer']}")

        if result.get("suggestions"):
            print("\n💡 Suggested follow-ups:")
            for s in result["suggestions"][:3]:
                print(f"   • {s}")


if __name__ == "__main__":
    run_cli()
