from retrieval.embeddings import EmbeddingModel


if __name__ == "__main__":

    model = EmbeddingModel()

    texts = [
        "What is the role of the AMF in the 5G System?",
        "The Access and Mobility Management Function (AMF).",
        "The Session Management Function (SMF).",
    ]

    embeddings = model.encode(texts)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])