from app.prompt_builder import PromptBuilder
from app.semantic_search import SemanticSearch


def main():

    question = "Tell me about bottle designs."

    search = SemanticSearch()

    results = search.search(question)

    builder = PromptBuilder()

    prompt = builder.build(question, results)

    print(prompt)


if __name__ == "__main__":
    main()