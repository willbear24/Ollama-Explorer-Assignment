import requests
import time

# Write a generate() helper function that calls Ollama’s /api/chat endpoint, measures response time, and returns the response text
# Run these 4 experiments

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"


def check_ollama():
    """Verify Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama is running. Available models: {models}")
        if MODEL not in models:
            print(
                f"ERROR: Model {MODEL!r} is not available. "
                f"Pull it with: ollama pull {MODEL}"
            )
            return False
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return False


def generate(prompt, system="You are a helpful assistant.", temperature=None):
    """Send a response to Ollama and return the response"""
    start = time.time()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
    )

    elapsed = time.time() - start
    if not response.ok:
        try:
            error = response.json().get("error", response.text)
        except requests.exceptions.JSONDecodeError:
            error = response.text
        raise RuntimeError(f"Ollama API error ({response.status_code}): {error}")

    data = response.json()
    message = data.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError(f"Unexpected Ollama response: {data!r}")
    text = message["content"]

    print(f"[Generated in {elapsed:.1f}s]")
    return text


# Experiment 1: Same question, different system prompts
# Ask "What is an API?" with three different system prompts: (a) no system prompt, (b) "Explain like I’m 5 years old," (c) "You are a senior software architect. Be technical and precise."
# Compare how the responses differ
system_prompts = [
    "",
    "Explain it like im a 5 year old.",
    "You are a senior software architect. Be technical and precise",
]

if __name__ == "__main__":
    if check_ollama():
        # TEST 1: 3 Different System Prompts, same question
        print("EXPERIMENT 1: Different system prompts, same question")
        question = "What is an API?"
        print(f"\nQuestion: {question}\n")

        for i, prompt in enumerate(system_prompts, start=1):
            label = prompt if prompt else "(No system prompt)"
            print("=" * 60)
            print(f"System Prompt {i}: {label}")
            print("=" * 60)
            response = generate(question, system=prompt)
            print(response)
            print("\n")

        # Comparison: No system prompt reads like a wikipedia article. The information is written clearly but technically, mostly in paragraph form but with numerated lists when applicable. The "5 year old" response breaks down the technical language into fun examples and tries to make it make sense with analogies. No technical language is used. The "Senior Architect", however, writes it out like a README file, even including the markdown language of a README. The information is divided in points and used deep vocabulary and dense information.

        # Experiment 2: RAG-style context grounding
        # Provide a short paragraph of context about a topic you choose
        # Ask a question that CAN be answered from the context
        # Ask a question that CANNOT be answered from the context
        # Does the model correctly refuse the second question?

        # TEST 2: RAG grounding with context
        print("=" * 60)
        print("EXPERIMENT 2: RAG-Style context grounding")
        print("=" * 60)

        context = (
            "JWT (JSON Web Token) authentication in FastAPI works by creating "
            "a /auth/token endpoint that validates credentials and returns a "
            "signed token. Protected endpoints verify the token on each request."
        )

        rag_sytem = (
            "Answer the users question based ONLY on the following context."
            "If the context doesn't contain the answer, say 'I don't have enough "
            "information to answer that.'\n\n"
            f"CONTEXT:\n{context}"
        )
        applicaple_question = "How does JWT Authentication in FastAPI?"
        irrelevant_question = "What is the best use case for Swagger UI?"
        positive_response = generate(applicaple_question, system=rag_sytem)
        negative_response = generate(irrelevant_question, system=rag_sytem)
        print(f"\nRelevant Query: {applicaple_question}")
        print(f"\nResponse: {positive_response}\n")
        print(f"\nIrrelevant Question: {irrelevant_question}")
        print(f"\nResponse: {negative_response}\n")

        # The model correctly refused the second query outside of the bounds of the context.

        # Experiment 3: Response timing
        # Ask questions of varying length (short: 5 words, medium: 20 words, long: 50 words)
        # Record and compare response times

        print("=" * 60)
        print("EXPERIMENT 3: Response Timing")
        print("=" * 60)

        questions = {
            "Short Question (5 words)": "What is a Python generator?",
            "Medium Question (20 words)": "How do list comprehensions differ from standard for-loops in Python in terms of syntax, readability, and overall execution performance?",
            "Long Question (50 words)": "When designing a production microservice architecture in Python, what key trade-offs should an engineering team consider between synchronous frameworks like Flask and asynchronous frameworks like FastAPI or Quart, specifically regarding connection concurrency, database I/O bottlenecks, overall memory footprint under heavy traffic loads, and the complexity of debugging async codebases?",
        }

        timing_results = []

        for label, question in questions.items():
            prompt_word_count = len(question.split())
            print(f"\n--- {label} ({prompt_word_count} words) ---")
            print(f"Prompt: {question}\n")

            start_time = time.time()
            response_text = generate(question)
            elapsed = time.time() - start_time
            response_word_count = len(response_text.split())

            timing_results.append(
                {
                    "label": label,
                    "prompt_words": prompt_word_count,
                    "elapsed": elapsed,
                    "response_words": response_word_count,
                }
            )

            print(f"\nResponse preview: {response_text[:150]}...")
            print(f"Time taken: {elapsed:.2f}s | Response words: {response_word_count}")

        # Comparison summary table
        print("\n" + "=" * 60)
        print("TIMING COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"{'Category':<28} {'Prompt Words':<14} {'Response Words':<16} {'Time (s)':<10}"
        )
        print("-" * 68)
        for r in timing_results:
            print(
                f"{r['label']:<28} {r['prompt_words']:<14} {r['response_words']:<16} {r['elapsed']:<10.2f}"
            )

        # Experiment 4: Temperature (optional)
        # Add "temperature": 0.1 to one request and "temperature": 1.0 to another with the same prompt
        # Compare how deterministic vs creative the responses are

        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Temperature Comparison")
        print("=" * 60)

        temp_prompt = "Come up with a description for a fantasy world with a decent amount of detail including landscape, architecture, races in the world, and a magic system. Limit yourself to 1000 characters please."
        print(f"Prompt: {temp_prompt}\n")

        # Running twice at 0.1 to see consistency, and twice at 1.0 to see variation
        test_temperatures = [0.1, 1.0]

        for temp in test_temperatures:
            print("-" * 60)
            print(f"Testing Temperature = {temp}")
            print("-" * 60)
            for attempt in range(1, 3):
                print(f"[Run {attempt} @ temp={temp}]")
                res = generate(temp_prompt, temperature=temp)
                print(res)
                print()

        # Conlcusion: I gave the system a very creative prompt. While both were creative in their response, the variation and more floral in it's language, showcasing the more "creative" nature of running a prompt with high temperature. The low temp run was still creative, but the sentence structure was all kind of similar and the vocabulary was much more precise and less interested in telling a story.
